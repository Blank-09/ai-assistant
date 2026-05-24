"""Voice Assistant — Entry point.

Starts a uvicorn server exposing the voice assistant as a WebRTC stream.
The frontend (Vite + React) connects via the browser's RTCPeerConnection.

Usage:
    uv run python main.py
    uv run python main.py --model gemma3:4b
    uv run python main.py --host 0.0.0.0 --port 7860
    uv run python main.py --verbose
"""

import argparse
import sys

import uvicorn
from loguru import logger

from voice_assistant.server import _make_app


def configure_logging(verbose: bool = False) -> None:
    """Configure loguru logging."""
    logger.remove()

    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    logger.add(
        ".logs/app.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="3 days",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice Assistant — WebRTC voice chat with local AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python main.py                        # Start with defaults
  uv run python main.py --model gemma3:4b      # Specify model
  uv run python main.py --host 0.0.0.0         # Expose on all interfaces
  uv run python main.py --port 8000            # Custom port
  uv run python main.py --verbose              # Enable debug logging
        """,
    )
    parser.add_argument(
        "--model",
        default="gemma3:4b",
        help="Ollama model name (default: gemma3:4b)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt for the assistant",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the server to (default: 7860)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    configure_logging(verbose=args.verbose)

    logger.info(
        "Starting Voice Assistant server (model={model}, {host}:{port})",
        model=args.model,
        host=args.host,
        port=args.port,
    )

    app = _make_app(model=args.model, system_prompt=args.system_prompt)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.verbose else "info",
    )


if __name__ == "__main__":
    main()
