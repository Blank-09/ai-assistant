"""Conversation history management with sliding window."""

from loguru import logger


class Conversation:
    """Manages conversation history with a sliding window.

    The system prompt is always pinned at position 0 and never evicted.
    The last `max_turns` user/assistant exchanges are kept in context.
    Older turns are silently dropped.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful voice assistant. Respond concisely and naturally, "
        "as if having a spoken conversation. Keep responses brief — typically "
        "1-3 sentences unless the user asks for detail. Avoid markdown formatting, "
        "bullet points, or code blocks since your responses will be spoken aloud."
    )

    def __init__(self, system_prompt: str | None = None, max_turns: int = 10) -> None:
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._max_turns = max_turns
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt}
        ]
        logger.info(
            "Conversation initialized with max_turns={max_turns}",
            max_turns=max_turns,
        )

    def add_user_message(self, text: str) -> None:
        """Append a user utterance to the history."""
        self._messages.append({"role": "user", "content": text})
        self._trim()
        logger.debug("User message added ({n} messages total)", n=len(self._messages))

    def add_assistant_message(self, text: str) -> None:
        """Append an assistant response to the history."""
        self._messages.append({"role": "assistant", "content": text})
        self._trim()
        logger.debug(
            "Assistant message added ({n} messages total)", n=len(self._messages)
        )

    def get_messages(self) -> list[dict[str, str]]:
        """Return the current message list for the LLM, including system prompt."""
        return list(self._messages)

    def _trim(self) -> None:
        """Keep only the system prompt + the last max_turns exchanges."""
        # Each turn = 1 user + 1 assistant = 2 messages
        max_history_messages = self._max_turns * 2
        history = self._messages[1:]  # Everything except system prompt

        if len(history) > max_history_messages:
            dropped = len(history) - max_history_messages
            history = history[-max_history_messages:]
            self._messages = [self._messages[0]] + history
            logger.info("Trimmed {dropped} old messages from history", dropped=dropped)

    def clear(self) -> None:
        """Reset history, keeping only the system prompt."""
        self._messages = [self._messages[0]]
        logger.info("Conversation history cleared")

    @property
    def turn_count(self) -> int:
        """Number of complete user/assistant exchanges."""
        history = self._messages[1:]
        return len(history) // 2
