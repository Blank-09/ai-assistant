"""Terminal UI for the voice assistant."""

from loguru import logger

class ConsoleUI:
    """Handles all terminal visual output and emojis."""
    
    def print_banner(self) -> None:
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

    def print_devices(self, device_text: str) -> None:
        print("\n  🎤 Audio devices:")
        for line in device_text.split("\n"):
            print(f"     {line}")

    def print_init_start(self) -> None:
        print("\n🚀 Initializing Voice Assistant...\n")
        
    def print_init_step(self, message: str) -> None:
        print(message)
        
    def print_init_complete(self) -> None:
        print("\n✅ All systems ready!\n")

    def show_listening(self) -> None:
        print("=" * 50)
        print("🎤 Listening... (speak naturally, Ctrl+C to quit)")
        print("=" * 50)
        
    def show_listening_short(self) -> None:
        print("\n🎤 Listening...")
        
    def show_transcribing(self) -> None:
        print("📝 Transcribing...")
        
    def show_user_message(self, text: str) -> None:
        print(f"💬 You: \"{text}\"")
        
    def show_thinking(self) -> None:
        print("🧠 Thinking...")
        print("🔊 Assistant: ", end="", flush=True)

    def print_assistant_sentence(self, sentence: str) -> None:
        print(f"{sentence} ", end="", flush=True)
        
    def show_assistant_done(self) -> None:
        print()  # Newline after streaming output
        
    def show_shutdown(self) -> None:
        print("\n\n👋 Shutting down gracefully...")

    def show_goodbye(self) -> None:
        print("\n👋 Goodbye!")
