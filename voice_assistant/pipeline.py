"""Pipeline orchestrator — wires all stages into a streaming voice loop.

The main loop:
  1. Listen for speech (VAD detects utterance boundaries)
  2. Transcribe the utterance (faster-whisper)
  3. Stream LLM response (Ollama), buffering tokens into sentences
  4. Synthesize each sentence as it completes (Kokoro TTS)
  5. Play audio chunks while the LLM continues generating
  6. Return to listening

Sentence-level streaming: the LLM→TTS→Speaker path overlaps so the user
hears the first sentence while the LLM is still generating the rest.
"""

import asyncio

from loguru import logger

from voice_assistant.audio import MicCapture, SpeakerPlayback, list_audio_devices
from voice_assistant.conversation import Conversation
from voice_assistant.llm import check_connection, generate_sentences
from voice_assistant.stt import SpeechToText
from voice_assistant.tts import TextToSpeech
from voice_assistant.vad import VoiceActivityDetector


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

        # Audio queues
        self._mic_queue: asyncio.Queue = asyncio.Queue()
        self._playback_queue: asyncio.Queue = asyncio.Queue()

    async def initialize(self) -> bool:
        """Load all models and verify connections. Returns True if successful."""
        print("\n🚀 Initializing Voice Assistant...\n")

        # Check Ollama connection
        print("  🔌 Checking Ollama connection...")
        if not await check_connection(self._model):
            print(f"  ❌ Ollama model '{self._model}' not available.")
            print("     Make sure Ollama is running and the model is pulled:")
            print(f"     ollama pull {self._model}")
            return False
        print(f"  ✅ Ollama connected — {self._model}")

        # Load VAD
        print("  🎯 Loading Voice Activity Detection...")
        self._vad.load()
        print("  ✅ Silero VAD loaded (CPU)")

        # Load STT
        print("  📝 Loading Speech-to-Text model...")
        self._stt.load()
        print("  ✅ faster-whisper 'small' loaded (GPU)")

        # Load TTS
        print("  🔊 Loading Text-to-Speech model...")
        self._tts.load()
        print("  ✅ Kokoro TTS loaded (GPU)")

        # List audio devices
        print("\n  🎤 Audio devices:")
        for line in list_audio_devices().split("\n"):
            print(f"     {line}")

        print("\n✅ All systems ready!\n")
        return True

    async def run(self) -> None:
        """Run the main voice loop. Blocks until interrupted."""
        # Start mic capture
        mic = MicCapture(self._mic_queue, device=self._mic_device)
        await mic.start()

        try:
            print("=" * 50)
            print("🎤 Listening... (speak naturally, Ctrl+C to quit)")
            print("=" * 50)

            async for utterance in self._vad.detect_utterances(self._mic_queue):
                await self._handle_turn(utterance)
                print("\n🎤 Listening...")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            await mic.stop()
            print("\n👋 Goodbye!")

    async def _handle_turn(self, utterance_audio) -> None:
        """Handle a single conversational turn.

        1. Transcribe the utterance
        2. Stream LLM response
        3. Synthesize and play audio for each sentence
        """
        # --- Transcribe ---
        print("📝 Transcribing...")
        text = await self._stt.transcribe(utterance_audio)

        if not text or text.isspace():
            logger.debug("Empty transcription, skipping turn")
            return

        print(f"💬 You: \"{text}\"")

        # --- Add to conversation ---
        self._conversation.add_user_message(text)

        # --- Stream LLM response ---
        print("🧠 Thinking...")

        full_response = []

        # Queues for the pipeline stages
        tts_queue: asyncio.Queue = asyncio.Queue()
        playback_queue: asyncio.Queue = asyncio.Queue()

        # Start background workers
        playback_task = asyncio.create_task(
            self._speaker.play_chunks(playback_queue)
        )
        tts_task = asyncio.create_task(
            self._tts_worker(tts_queue, playback_queue)
        )

        print("🔊 Assistant: ", end="", flush=True)

        async for sentence in generate_sentences(
            self._conversation.get_messages(), model=self._model
        ):
            full_response.append(sentence)
            print(f"{sentence} ", end="", flush=True)
            await tts_queue.put(sentence)

        print()  # Newline after streaming output

        # Signal end of response to TTS worker
        await tts_queue.put(None)

        # Wait for TTS and Playback to finish
        await tts_task
        await playback_task

        # Add complete response to conversation history
        complete_response = " ".join(full_response)
        self._conversation.add_assistant_message(complete_response)

        logger.info(
            "Turn complete. History: {n} turns",
            n=self._conversation.turn_count,
        )

    async def _tts_worker(
        self, tts_queue: asyncio.Queue, playback_queue: asyncio.Queue
    ) -> None:
        """Consume sentences, synthesize audio, and queue for playback."""
        while True:
            text = await tts_queue.get()
            if text is None:
                # End of response — signal playback worker to stop
                await playback_queue.put(None)
                break
            
            try:
                audio = await self._tts.synthesize(text)
                if len(audio) > 0:
                    await playback_queue.put(audio)
            except Exception as e:
                logger.error("TTS synthesis failed for \"{text}\": {err}", text=text[:50], err=e)
