# WHY THIS FILE EXISTS:
# Defines the abstract base class (interface) for all LLM providers.
# By programming to an interface, the rest of the codebase doesn't care
# whether it's talking to OpenAI, Azure OpenAI, Anthropic, or a mock.
# Swapping providers = swap one class, zero other changes.

from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional

from app.models.chat import ChatMessage, TokenUsage


class LLMResponse:
    """Value object returned by all LLM providers."""
    def __init__(self, content: str, usage: TokenUsage, model: str, latency_ms: float):
        self.content = content
        self.usage = usage
        self.model = model
        self.latency_ms = latency_ms

    def __repr__(self) -> str:
        return (
            f"LLMResponse(model={self.model!r}, "
            f"tokens={self.usage.total_tokens}, "
            f"latency_ms={self.latency_ms})"
        )


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.

    Concrete implementations must override `complete()` and `stream()`.
    Optional: override `health_check()` for readiness probes.

    Why abstract methods?
    - Python's ABC raises NotImplementedError at class instantiation time
      if a required method is not implemented. This catches mistakes early.
    """

    @abstractmethod
    async def complete(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Generate a completion for the given conversation.

        Args:
            messages: Conversation history (system + user + assistant turns).
            max_tokens: Maximum tokens in the response (None = use default).
            temperature: Sampling temperature 0–2. Lower = more deterministic.

        Returns:
            LLMResponse with the generated content and usage stats.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens as they are generated.

        Yields:
            Token strings as they arrive from the LLM.
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if the LLM provider is reachable.
        Default implementation: always True (override for real providers).
        """
        return True

    @property
    def provider_name(self) -> str:
        """Human-readable provider name for logging."""
        return self.__class__.__name__
