"""Session handler — one instance per WebRTC connection.

Each Session owns an independent Conversation (sliding window history).
STT and TTS models are shared across sessions and injected at construction.

FastRTC calls StreamHandler.copy() at the start of each new WebRTC connection,
which returns a fresh SessionHandler with its own Conversation — this is the
per-session isolation mechanism. No external factory needed.

The handler is called by FastRTC's ReplyOnPause when a complete Utterance
is detected. It transcribes, queries the LLM, synthesizes sentence chunks,
and yields resampled audio frames back to the browser.

Sample rate note:
    Kokoro TTS outputs at 24kHz. WebRTC requires a consistent sample rate.
    WEBRTC_SAMPLE_RATE is set to 48kHz (standard WebRTC default) but can be
    changed to 16000 if 16kHz is preferred for your WebRTC configuration.
"""

import asyncio
import queue
import threading
from math import gcd

import numpy as np
from loguru import logger
from scipy.signal import resample_poly

from voice_assistant.core.conversation import Conversation
from voice_assistant.models.llm import generate_sentences
from voice_assistant.models.stt import SpeechToText
from voice_assistant.models.tts import TextToSpeech, SAMPLE_RATE as TTS_SAMPLE_RATE

# WebRTC output sample rate.
# Can be changed to 16_000 if your WebRTC setup prefers 16kHz.
WEBRTC_SAMPLE_RATE = 48_000


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    g = gcd(src_rate, dst_rate)
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)


class SessionHandler:
    """Handles one conversational turn for a single WebRTC Session.

    FastRTC's ReplyOnPause calls this instance when an Utterance is detected.
    The handler is stateful — it accumulates Turns into its own Conversation.

    Args:
        model: Ollama model name.
        stt: Shared SpeechToText instance (loaded once at startup).
        tts: Shared TextToSpeech instance (loaded once at startup).
        system_prompt: Optional custom system prompt for this session.
    """

    def __init__(
        self,
        model: str,
        stt: SpeechToText,
        tts: TextToSpeech,
        system_prompt: str | None = None,
    ) -> None:
        self._model = model
        self._stt = stt
        self._tts = tts
        self._conversation = Conversation(system_prompt=system_prompt)
        logger.info("Session started (model={model})", model=model)

    def copy(self) -> "SessionHandler":
        """Called by FastRTC per new WebRTC connection.

        Returns a fresh SessionHandler with its own Conversation,
        sharing the same (already-loaded) STT and TTS model instances.
        """
        return SessionHandler(
            model=self._model,
            stt=self._stt,
            tts=self._tts,
            system_prompt=None,  # Use default; can be parameterised later
        )

    def __call__(self, audio: tuple[int, np.ndarray]):
        """Handle one Utterance: transcribe → LLM → TTS → yield audio frames.

        FastRTC's ReplyOnPause calls next() on the returned generator, so this
        must be a sync generator. The async pipeline runs in a daemon thread
        with its own event loop and feeds chunks via a queue.

        Args:
            audio: (sample_rate, samples) tuple provided by FastRTC.

        Yields:
            (WEBRTC_SAMPLE_RATE, audio_chunk) tuples for FastRTC to send back.
        """
        sample_rate, samples = audio

        # (channels, samples) → (samples,) mono
        samples = np.squeeze(samples)
        if samples.ndim > 1:
            samples = samples.mean(axis=0)

        # Normalize to float32 [-1, 1] — WebRTC delivers int16 PCM
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        else:
            samples = samples.astype(np.float32)

        # Whisper requires 16 kHz input
        stt_samples = _resample(samples, sample_rate, 16_000)

        chunk_queue: queue.Queue = queue.Queue()
        _DONE = object()

        async def _run() -> None:
            logger.debug(
                "Transcribing utterance ({sr}Hz→16kHz, {n} samples)",
                sr=sample_rate,
                n=len(stt_samples),
            )
            text = await self._stt.transcribe(stt_samples)

            if not text or text.isspace():
                logger.debug("Empty transcription, skipping turn")
                chunk_queue.put(_DONE)
                return

            logger.info("User: {text}", text=text)
            self._conversation.add_user_message(text)

            full_response: list[str] = []

            async for sentence in generate_sentences(
                self._conversation.get_messages(), model=self._model
            ):
                full_response.append(sentence)
                logger.debug("Assistant sentence: {s}", s=sentence)

                async for chunk in self._tts.synthesize_streaming(sentence):
                    resampled = _resample(chunk, TTS_SAMPLE_RATE, WEBRTC_SAMPLE_RATE)
                    chunk_queue.put((WEBRTC_SAMPLE_RATE, resampled))

            complete_response = " ".join(full_response)
            self._conversation.add_assistant_message(complete_response)
            logger.info(
                "Turn complete ({n} turns in history)",
                n=self._conversation.turn_count,
            )
            chunk_queue.put(_DONE)

        threading.Thread(target=asyncio.run, args=(_run(),), daemon=True).start()

        while True:
            item = chunk_queue.get()
            if item is _DONE:
                break
            yield item
