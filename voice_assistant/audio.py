"""Audio I/O — microphone capture and speaker playback via sounddevice.

Mic capture pushes raw PCM frames into an asyncio queue.
Speaker playback consumes audio chunks from an asyncio queue.
Both are async-queue-based for clean decoupling from the pipeline.
"""

import asyncio
from typing import Optional

import numpy as np
import sounddevice as sd
from loguru import logger


# Audio format constants
SAMPLE_RATE_INPUT = 16000  # 16kHz for VAD and STT
SAMPLE_RATE_OUTPUT = 24000  # 24kHz for Kokoro TTS output
CHANNELS = 1  # Mono
DTYPE_INPUT = "int16"
BLOCK_SIZE_INPUT = 512  # 32ms at 16kHz — matches Silero VAD chunk size


class MicCapture:
    """Captures audio from the microphone and pushes frames to an async queue.

    The sounddevice callback runs in a PortAudio thread. We bridge into
    asyncio land using loop.call_soon_threadsafe.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        device: Optional[int] = None,
        sample_rate: int = SAMPLE_RATE_INPUT,
        block_size: int = BLOCK_SIZE_INPUT,
    ) -> None:
        self._queue = queue
        self._device = device
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._stream: Optional[sd.InputStream] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags
    ) -> None:
        """Called from the PortAudio thread for each audio block."""
        if status:
            logger.warning("Audio input status: {status}", status=status)

        # Copy the data — indata buffer is reused by sounddevice
        audio_chunk = indata[:, 0].copy()  # (frames,) mono float32
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, audio_chunk)

    async def start(self) -> None:
        """Open the microphone stream and begin capturing."""
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=self._block_size,
            device=self._device,
            channels=CHANNELS,
            dtype="float32",  # sounddevice gives us float32, VAD expects float32
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info(
            "Mic capture started (rate={rate}Hz, blocksize={bs}, device={dev})",
            rate=self._sample_rate,
            bs=self._block_size,
            dev=self._device or "default",
        )

    async def stop(self) -> None:
        """Stop and close the microphone stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Mic capture stopped")


class SpeakerPlayback:
    """Plays audio chunks from an async queue through the speakers.

    Consumes numpy arrays (float32, 24kHz mono) from the queue and plays
    them sequentially. Signals completion when a None sentinel is received.
    """

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = SAMPLE_RATE_OUTPUT,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate

    async def play_chunks(self, audio_queue: asyncio.Queue) -> None:
        """Play audio chunks from the queue until a None sentinel is received.

        Each chunk is a numpy float32 array. A None value signals end of response.
        """
        loop = asyncio.get_running_loop()

        # Open a single continuous stream to prevent PortAudio errors
        # caused by repeatedly re-opening streams between sentences.
        try:
            stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                device=self._device,
            )
            stream.start()

            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    logger.debug("Speaker received end-of-response sentinel")
                    break

                logger.debug("Playing audio chunk: {n} samples", n=len(chunk))
                # stream.write is blocking — run in executor
                await loop.run_in_executor(None, self._play_sync, stream, chunk)
                
        finally:
            stream.stop()
            stream.close()

    def _play_sync(self, stream: sd.OutputStream, audio: np.ndarray) -> None:
        """Synchronously write an audio array to the open stream."""
        stream.write(audio)


def list_audio_devices() -> str:
    """Return a formatted string of available audio devices."""
    return str(sd.query_devices())
