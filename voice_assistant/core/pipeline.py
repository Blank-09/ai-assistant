"""Pipeline orchestrator — wires all stages into a streaming voice loop.

The main loop:
  1. Listen for speech (VAD detects utterance boundaries)
  2. Transcribe the utterance (faster-whisper)
  3. Stream LLM response (Ollama), yielding complete sentences
  4. Pushes sentences to PlaybackStream
  5. Return to listening

Sentence-level streaming: the LLM→TTS→Speaker path overlaps so the user
hears the first sentence while the LLM is still generating the rest.
"""

import asyncio

from loguru import logger

from voice_assistant.io.audio import MicCapture, SpeakerPlayback, list_audio_devices
from voice_assistant.core.conversation import Conversation
from voice_assistant.models.llm import check_connection, generate_sentences
from voice_assistant.models.stt import SpeechToText
from voice_assistant.models.tts import TextToSpeech
from voice_assistant.models.vad import VoiceActivityDetector
from voice_assistant.io.ui import ConsoleUI
from voice_assistant.io.playback import PlaybackStream


class VoicePipeline:
    """Orchestrates the full voice assistant pipeline."""

    def __init__(
        self,
        model: str = "gemma3:4b",
        mic_device: int | None = None,
        speaker_device: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._model = model
        self._mic_device = mic_device
        self._speaker_device = speaker_device

        # Components
        self._vad = VoiceActivityDetector()
        self._stt = SpeechToText()
        self._tts = TextToSpeech()
        self._conversation = Conversation(system_prompt=system_prompt)
        self._speaker = SpeakerPlayback(device=speaker_device)
        self._ui = ConsoleUI()
        self._playback_stream = PlaybackStream(self._tts, self._speaker)

        # Audio queues
        self._mic_queue: asyncio.Queue = asyncio.Queue()

    async def initialize(self) -> bool:
        """Load all models and verify connections. Returns True if successful."""
        self._ui.print_init_start()

        # Check Ollama connection
        self._ui.print_init_step("  🔌 Checking Ollama connection...")
        if not await check_connection(self._model):
            self._ui.print_init_step(f"  ❌ Ollama model '{self._model}' not available.")
            self._ui.print_init_step("     Make sure Ollama is running and the model is pulled:")
            self._ui.print_init_step(f"     ollama pull {self._model}")
            return False
        self._ui.print_init_step(f"  ✅ Ollama connected — {self._model}")

        # Load VAD
        self._ui.print_init_step("  🎯 Loading Voice Activity Detection...")
        self._vad.load()
        self._ui.print_init_step("  ✅ Silero VAD loaded (CPU)")

        # Load STT
        self._ui.print_init_step("  📝 Loading Speech-to-Text model...")
        self._stt.load()
        self._ui.print_init_step("  ✅ faster-whisper 'small' loaded (GPU)")

        # Load TTS
        self._ui.print_init_step("  🔊 Loading Text-to-Speech model...")
        self._tts.load()
        self._ui.print_init_step("  ✅ Kokoro TTS loaded (GPU)")

        # List audio devices
        self._ui.print_devices(list_audio_devices())

        self._ui.print_init_complete()
        return True

    async def run(self) -> None:
        """Run the main voice loop. Blocks until interrupted."""
        # Start mic capture
        mic = MicCapture(self._mic_queue, device=self._mic_device)
        await mic.start()

        try:
            self._ui.show_listening()

            async for utterance in self._vad.detect_utterances(self._mic_queue):
                await self._handle_turn(utterance)
                self._ui.show_listening_short()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            await mic.stop()
            self._ui.show_goodbye()

    async def _handle_turn(self, utterance_audio) -> None:
        """Handle a single conversational turn.

        1. Transcribe the utterance
        2. Stream LLM response
        3. Push to PlaybackStream
        """
        # --- Transcribe ---
        self._ui.show_transcribing()
        text = await self._stt.transcribe(utterance_audio)

        if not text or text.isspace():
            logger.debug("Empty transcription, skipping turn")
            return

        self._ui.show_user_message(text)

        # --- Add to conversation ---
        self._conversation.add_user_message(text)

        # --- Stream LLM response ---
        self._ui.show_thinking()

        await self._playback_stream.start()

        full_response = []

        async for sentence in generate_sentences(
            self._conversation.get_messages(), model=self._model
        ):
            full_response.append(sentence)
            self._ui.print_assistant_sentence(sentence)
            await self._playback_stream.enqueue_sentence(sentence)

        self._ui.show_assistant_done()

        # Wait for TTS and Playback to finish
        await self._playback_stream.wait_for_completion()

        # Add complete response to conversation history
        complete_response = " ".join(full_response)
        self._conversation.add_assistant_message(complete_response)

        logger.info(
            "Turn complete. History: {n} turns",
            n=self._conversation.turn_count,
        )
