# WHY THIS FILE EXISTS:
# Minimal in-request conversation memory for the agent.
# "In-request" means the memory lives only for the duration of one /agents/run call.
# This is intentional: simpler, no persistence layer needed, no PII retention risk.
# For persistent memory across sessions, add a Redis/DB backend here.

from collections import deque
from typing import Deque, Dict, List

from app.core.config import get_settings


class ConversationMemory:
    """
    Fixed-size sliding window of conversation turns.

    Uses a deque (double-ended queue) with maxlen.
    When the deque is full and a new item is added, the oldest item
    is automatically removed from the left — no manual management needed.

    This is the simplest possible memory implementation: no summarization,
    no semantic compression, no external storage. For a production chatbot,
    replace this with a summarizing memory or a Redis-backed store.
    """

    def __init__(self, max_turns: int | None = None):
        settings = get_settings()
        limit = max_turns or settings.agent_memory_turns
        # maxlen=limit*2 because each "turn" = 1 user msg + 1 assistant msg
        self._turns: Deque[Dict[str, str]] = deque(maxlen=limit * 2)

    def add_user_message(self, content: str) -> None:
        """Record a user message."""
        self._turns.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Record an assistant (agent) response."""
        self._turns.append({"role": "assistant", "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """Return conversation history as a list of role/content dicts."""
        return list(self._turns)

    def load_from_history(self, history: List[Dict[str, str]]) -> None:
        """Pre-populate memory from an existing history (e.g. from the request body)."""
        for turn in history:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                self._turns.append({"role": turn["role"], "content": turn["content"]})

    def clear(self) -> None:
        """Reset the memory."""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
