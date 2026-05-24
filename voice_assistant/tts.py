"""Text-to-Speech using Kokoro.

Loads the Kokoro model on GPU and synthesizes text into audio chunks.
Uses KPipeline's generator-based streaming — each call to synthesize()
yields audio chunks as they're produced, enabling sentence-level streaming
in the pipeline.
"""

import asyncio
from collections.abc import AsyncGenerator
from functools import partial

import numpy as np
import torch
from kokoro import KPipeline
from loguru import logger


DEFAULT_VOICE = "af_heart"  # Warm, conversational female voice
DEFAULT_SPEED = 1.0
DEFAULT_LANG = "a"  # American English
SAMPLE_RATE = 24000  # Kokoro outputs at 24kHz


class TextToSpeech:
    """Synthesizes text to audio using Kokoro TTS.

    Loads the model once at startup. Synthesis runs in a thread pool
    to avoid blocking the event loop.
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_SPEED,
        lang: str = DEFAULT_LANG,
    ) -> None:
        self._voice = voice
        self._speed = speed
        self._lang = lang
        self._pipeline: KPipeline | None = None

    def load(self) -> None:
        """Load the Kokoro TTS pipeline. Call once at startup."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading Kokoro TTS (lang={lang}, voice={voice}, device={device})...",
            lang=self._lang,
            voice=self._voice,
            device=device,
        )
        if device == "cpu":
            logger.warning("CUDA not available for TTS — falling back to CPU (slower)")
        self._pipeline = KPipeline(lang_code=self._lang, device=device)
        logger.info("Kokoro TTS loaded successfully on {device}", device=device)

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text into a single audio array.

        Args:
            text: Text to synthesize.

        Returns:
            numpy float32 array of audio samples at 24kHz.
        """
        if self._pipeline is None:
            raise RuntimeError("TTS model not loaded. Call load() first.")

        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(
            None, partial(self._synthesize_sync, text)
        )
        return audio

    async def synthesize_streaming(
        self, text: str
    ) -> AsyncGenerator[np.ndarray, None]:
        """Synthesize text and yield audio chunks as they're produced.

        Kokoro's KPipeline returns a generator — we bridge it to async
        by running each chunk's synthesis in the executor.

        Args:
            text: Text to synthesize.

        Yields:
            numpy float32 arrays of audio chunks at 24kHz.
        """
        if self._pipeline is None:
            raise RuntimeError("TTS model not loaded. Call load() first.")

        loop = asyncio.get_running_loop()

        # Get the generator from KPipeline (runs in executor since it may
        # do initial processing)
        generator = await loop.run_in_executor(
            None,
            partial(
                self._pipeline,
                text,
                voice=self._voice,
                speed=self._speed,
            ),
        )

        # Iterate over chunks — each next() call may do GPU work
        for graphemes, phonemes, audio in generator:
            if audio is not None and len(audio) > 0:
                duration_ms = len(audio) / SAMPLE_RATE * 1000
                logger.debug(
                    "TTS chunk: \"{text}\" -> {dur:.0f}ms audio",
                    text=graphemes[:50],
                    dur=duration_ms,
                )
                yield audio

    def _synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous full synthesis — runs in a thread pool."""
        chunks = []
        for graphemes, phonemes, audio in self._pipeline(
            text, voice=self._voice, speed=self._speed
        ):
            if audio is not None and len(audio) > 0:
                chunks.append(audio)

        if not chunks:
            logger.warning("TTS produced no audio for: \"{text}\"", text=text[:80])
            return np.array([], dtype=np.float32)

        result = np.concatenate(chunks)
        duration_ms = len(result) / SAMPLE_RATE * 1000
        logger.info(
            "TTS synthesized: \"{text}\" -> {dur:.0f}ms audio",
            text=text[:80],
            dur=duration_ms,
        )
        return result

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return SAMPLE_RATE
