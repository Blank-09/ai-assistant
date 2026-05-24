"""Voice Activity Detection using Silero VAD.

Consumes raw audio chunks from a mic queue and yields complete utterances
(continuous speech bounded by silence on both sides).

Uses VADIterator for streaming chunk-by-chunk detection with 512-sample
(32ms) frames at 16kHz.
"""

import asyncio
from collections.abc import AsyncGenerator

import numpy as np
import torch
from loguru import logger
from silero_vad import VADIterator, load_silero_vad


# VAD configuration
VAD_THRESHOLD = 0.5  # Speech probability threshold
MIN_SILENCE_MS = 600  # Silence duration (ms) to consider speech ended
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # 32ms at 16kHz — required by Silero VAD v5+


class VoiceActivityDetector:
    """Detects speech boundaries in a stream of audio chunks.

    Wraps Silero VAD's VADIterator to provide an async generator that yields
    complete utterances (numpy arrays of continuous speech).
    """

    def __init__(
        self,
        threshold: float = VAD_THRESHOLD,
        min_silence_ms: int = MIN_SILENCE_MS,
    ) -> None:
        self._threshold = threshold
        self._min_silence_ms = min_silence_ms
        self._model = None
        self._vad_iterator = None

    def load(self) -> None:
        """Load the Silero VAD model. Call once at startup."""
        torch.set_num_threads(1)  # Recommended for Silero VAD performance
        self._model = load_silero_vad()
        self._vad_iterator = VADIterator(
            self._model,
            threshold=self._threshold,
            min_silence_duration_ms=self._min_silence_ms,
            sampling_rate=SAMPLE_RATE,
        )
        logger.info(
            "Silero VAD loaded (threshold={th}, min_silence={ms}ms)",
            th=self._threshold,
            ms=self._min_silence_ms,
        )

    async def detect_utterances(
        self, mic_queue: asyncio.Queue
    ) -> AsyncGenerator[np.ndarray, None]:
        """Yield complete utterances from the mic audio stream.

        Listens to audio chunks from the mic queue, detects speech onset and
        offset, and yields the accumulated audio buffer for each utterance.

        Args:
            mic_queue: Queue of numpy float32 audio chunks (512 samples each).

        Yields:
            numpy float32 arrays containing complete utterances.
        """
        if self._vad_iterator is None:
            raise RuntimeError("VAD not loaded. Call load() first.")

        audio_buffer: list[np.ndarray] = []
        is_speaking = False

        while True:
            chunk = await mic_queue.get()

            # Convert numpy to torch tensor for Silero
            tensor = torch.from_numpy(chunk).float()
            speech_dict = self._vad_iterator(tensor, return_seconds=False)

            if speech_dict is not None:
                if "start" in speech_dict:
                    # Speech onset detected
                    is_speaking = True
                    audio_buffer.clear()
                    audio_buffer.append(chunk)
                    logger.debug("Speech started")

                elif "end" in speech_dict:
                    # Speech offset detected — yield the complete utterance
                    audio_buffer.append(chunk)
                    is_speaking = False

                    utterance = np.concatenate(audio_buffer)
                    duration_ms = len(utterance) / SAMPLE_RATE * 1000
                    logger.info(
                        "Utterance complete: {dur:.0f}ms, {samples} samples",
                        dur=duration_ms,
                        samples=len(utterance),
                    )

                    audio_buffer.clear()
                    self._vad_iterator.reset_states()
                    yield utterance

            elif is_speaking:
                # Mid-speech — keep accumulating
                audio_buffer.append(chunk)

    def reset(self) -> None:
        """Reset the VAD state for a new conversation."""
        if self._vad_iterator is not None:
            self._vad_iterator.reset_states()
            logger.debug("VAD state reset")
