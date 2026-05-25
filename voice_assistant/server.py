"""FastAPI + FastRTC server.

Exposes the voice assistant as a WebRTC audio stream. Each browser connection
gets its own Session (via StreamHandler.copy()), so multiple users can run
independent conversations concurrently.

Models (STT, TTS) are loaded once at startup and shared across all sessions.
Each Session owns an independent Conversation instance (managed by SessionHandler).

Endpoints added by FastRTC (via stream.mount):
    POST /webrtc/offer   — WebRTC SDP offer exchange
    GET  /webrtc/offer   — same, for GET-based handshake
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastrtc import AlgoOptions, ReplyOnPause, SileroVadOptions, Stream
from loguru import logger

from voice_assistant.core.session import SessionHandler
from voice_assistant.models.llm import check_connection
from voice_assistant.models.stt import SpeechToText
from voice_assistant.models.tts import TextToSpeech

# Module-level model singletons — loaded once, shared across sessions.
_stt: SpeechToText | None = None
_tts: TextToSpeech | None = None
_model: str = "gemma3:4b"
_system_prompt: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, then mount the WebRTC stream, release at shutdown."""
    global _stt, _tts

    logger.info("Loading models...")

    if not await check_connection(_model):
        raise RuntimeError(
            f"Ollama model '{_model}' not available. "
            f"Make sure Ollama is running and the model is pulled: ollama pull {_model}"
        )
    logger.info("Ollama connected — {model}", model=_model)

    _stt = SpeechToText()
    _stt.load()
    logger.info("STT loaded")

    _tts = TextToSpeech()
    _tts.load()
    logger.info("TTS loaded")

    # Build handler and stream here so they receive fully-loaded model instances.
    # Constructing them in _make_app() would capture the None globals.
    handler = SessionHandler(
        model=_model,
        stt=_stt,
        tts=_tts,
        system_prompt=_system_prompt,
    )
    stream = Stream(
        handler=ReplyOnPause(
            handler,
            algo_options=AlgoOptions(
                audio_chunk_duration=0.6,
                started_talking_threshold=0.2,
                # Default 0.1 fires on brief mid-sentence pauses (breath, hesitation).
                # 0.02 requires the 0.6s chunk to be almost entirely silent.
                speech_threshold=0.02,
            ),
            model_options=SileroVadOptions(
                # Higher probability threshold → fewer acoustic dips classified as silence.
                threshold=0.6,
                min_silence_duration_ms=500,
            ),
        ),
        modality="audio",
        mode="send-receive",
    )
    stream.mount(app)

    logger.info("All models ready. Server accepting connections.")
    yield
    logger.info("Shutting down")


def _make_app(model: str, system_prompt: str | None) -> FastAPI:
    """Construct the FastAPI app with FastRTC stream mounted.

    Args:
        model: Ollama model name.
        system_prompt: Optional system prompt override for all sessions.

    Returns:
        Configured FastAPI application.
    """
    global _model, _system_prompt
    _model = model
    _system_prompt = system_prompt

    app = FastAPI(
        title="Voice Assistant",
        description="Real-time voice assistant via WebRTC",
        lifespan=lifespan,
    )

    # Allow the Vite dev server (and any origin in development) to connect.
    # Restrict origins appropriately in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app
