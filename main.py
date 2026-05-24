"""Voice Assistant — Entry point.

A real-time voice chat application powered by local AI models.
Speak naturally and get spoken responses from gemma3:4b via Ollama.

Usage:
    uv run python main.py
    uv run python main.py --model gemma3:4b
    uv run python main.py --list-devices
"""

import argparse
import asyncio
import sys

from loguru import logger

from voice_assistant.audio import list_audio_devices
from voice_assistant.pipeline import VoicePipeline


def configure_logging(verbose: bool = False) -> None:
    """Configure loguru logging."""
    logger.remove()  # Remove default handler

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

    # Also log to file for debugging
    logger.add(
        "voice_assistant.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="3 days",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice Assistant — Real-time voice chat with local AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python main.py                     # Start with defaults
  uv run python main.py --model gemma3:4b   # Specify model
  uv run python main.py --verbose           # Enable debug logging
  uv run python main.py --list-devices      # Show audio devices
        """,
    )
    parser.add_argument(
        "--model",
        default="gemma3:4b",
        help="Ollama model name (default: gemma3:4b)",
    )
    parser.add_argument(
        "--mic-device",
        type=int,
        default=None,
        help="Microphone device index (default: system default)",
    )
    parser.add_argument(
        "--speaker-device",
        type=int,
        default=None,
        help="Speaker device index (default: system default)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt for the assistant",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    return parser.parse_args()


def print_banner() -> None:
    """Print the startup banner."""
    print(
        """
╔══════════════════════════════════════════════════╗
║         🎙️  Voice Assistant v0.1.0  🎙️          ║
║                                                  ║
║  Real-time voice chat with local AI models       ║
║  STT: faster-whisper (small) | LLM: Ollama       ║
║  TTS: Kokoro | VAD: Silero                       ║
║                                                  ║
║  Press Ctrl+C to quit                            ║
╚══════════════════════════════════════════════════╝
"""
    )


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Handle --list-devices
    if args.list_devices:
        print("Available audio devices:\n")
        print(list_audio_devices())
        sys.exit(0)

    configure_logging(verbose=args.verbose)
    print_banner()

    logger.info(
        "Starting Voice Assistant (model={model})",
        model=args.model,
    )

    # Create and run the pipeline
    pipeline = VoicePipeline(
        model=args.model,
        mic_device=args.mic_device,
        speaker_device=args.speaker_device,
        system_prompt=args.system_prompt,
    )

    try:
        asyncio.run(_run_pipeline(pipeline))
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        logger.info("Shutdown complete")


async def _run_pipeline(pipeline: VoicePipeline) -> None:
    """Initialize and run the pipeline."""
    if not await pipeline.initialize():
        logger.error("Failed to initialize. Exiting.")
        sys.exit(1)

    await pipeline.run()


if __name__ == "__main__":
    main()
