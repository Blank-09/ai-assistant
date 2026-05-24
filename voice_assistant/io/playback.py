"""Playback stream combining TTS synthesis and speaker output."""

import asyncio
from loguru import logger

from voice_assistant.io.audio import SpeakerPlayback
from voice_assistant.models.tts import TextToSpeech

class PlaybackStream:
    """Encapsulates TTS synthesis and audio playback into a single stream."""
    
    def __init__(self, tts: TextToSpeech, speaker: SpeakerPlayback):
        self._tts = tts
        self._speaker = speaker
        self._tts_queue = asyncio.Queue()
        self._playback_queue = asyncio.Queue()
        self._tts_task = None
        self._playback_task = None

    async def start(self) -> None:
        """Start the background worker tasks for the stream."""
        self._tts_queue = asyncio.Queue()
        self._playback_queue = asyncio.Queue()
        
        self._playback_task = asyncio.create_task(
            self._speaker.play_chunks(self._playback_queue)
        )
        self._tts_task = asyncio.create_task(
            self._tts_worker(self._tts_queue, self._playback_queue)
        )
        logger.debug("PlaybackStream started")

    async def enqueue_sentence(self, sentence: str) -> None:
        """Add a sentence to be synthesized and played."""
        await self._tts_queue.put(sentence)

    async def wait_for_completion(self) -> None:
        """Signal the end of the stream and wait for all audio to finish playing."""
        await self._tts_queue.put(None)
        
        if self._tts_task:
            await self._tts_task
        if self._playback_task:
            await self._playback_task
        logger.debug("PlaybackStream completed")

    async def _tts_worker(self, tts_queue: asyncio.Queue, playback_queue: asyncio.Queue) -> None:
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
