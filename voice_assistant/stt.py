"""Speech-to-Text using faster-whisper.

Loads the Whisper 'small' model on GPU and transcribes complete utterance
audio buffers. Runs inference in a thread pool executor to avoid blocking
the asyncio event loop.
"""

import asyncio
from functools import partial

import numpy as np
import torch
from faster_whisper import WhisperModel
from loguru import logger


DEFAULT_MODEL_SIZE = "small"
DEFAULT_DEVICE = "auto"  # Will auto-detect CUDA
DEFAULT_COMPUTE_TYPE = "auto"  # float16 for CUDA, int8 for CPU


class SpeechToText:
    """Transcribes audio using faster-whisper.

    Loads the model once at startup, then transcribes utterances on demand.
    Inference runs in a thread pool to keep the event loop responsive.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: WhisperModel | None = None

    def load(self) -> None:
        """Load the Whisper model. Call once at startup."""
        logger.info(
            "Loading faster-whisper model '{size}' on {device} ({compute})...",
            size=self._model_size,
            device=self._device,
            compute=self._compute_type,
        )
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("faster-whisper model loaded successfully")

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe an audio utterance to text.

        Args:
            audio: numpy float32 array of audio samples at 16kHz mono.

        Returns:
            The transcribed text string.
        """
        if self._model is None:
            raise RuntimeError("STT model not loaded. Call load() first.")

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, partial(self._transcribe_sync, audio))
        return text

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription — runs in a thread pool."""
        segments, info = self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=False,  # We handle VAD ourselves
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        text = " ".join(text_parts).strip()
        logger.info(
            "Transcribed ({dur:.1f}s audio, {lang} p={prob:.0%}): \"{text}\"",
            dur=info.duration,
            lang=info.language,
            prob=info.language_probability,
            text=text,
        )
        return text
