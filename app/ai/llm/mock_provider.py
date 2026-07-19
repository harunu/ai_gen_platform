# WHY THIS FILE EXISTS:
# A deterministic mock LLM for local development and testing.
# "Deterministic" means the same input always produces the same output —
# tests don't flap, no API costs, no rate limits, no internet required.
# This is the DEFAULT provider (USE_MOCK_LLM=true in .env.example).

import asyncio
import time
from typing import AsyncIterator, List, Optional

from app.ai.llm.base import BaseLLM, LLMResponse
from app.core.logging import get_logger
from app.models.chat import ChatMessage, TokenUsage
from app.utils.text import estimate_tokens

logger = get_logger(__name__)

# Canned responses keyed by keywords in the last user message.
# This makes the mock useful for RAG testing — ask about RAG, get a RAG answer.
CANNED_RESPONSES = {
    "rag": (
        "Retrieval-Augmented Generation (RAG) combines a retrieval system with an LLM. "
        "Instead of relying solely on parametric knowledge baked into the model weights, "
        "RAG retrieves relevant documents from an external knowledge base at query time "
        "and provides them as context to the LLM. This grounds the response in up-to-date, "
        "verifiable facts and dramatically reduces hallucination for knowledge-intensive tasks."
    ),
    "embedding": (
        "Embeddings are dense vector representations of text. "
        "Semantically similar texts produce vectors that are close together in vector space, "
        "as measured by cosine similarity or dot product. "
        "In a RAG system, both documents and queries are embedded into the same vector space "
        "so that relevant documents can be found via nearest-neighbor search."
    ),
    "fastapi": (
        "FastAPI is a modern Python web framework built on Starlette and Pydantic. "
        "It generates OpenAPI documentation automatically from type annotations, "
        "supports full async/await, and validates request/response shapes via Pydantic. "
        "Its dependency injection system makes it easy to share resources (DB connections, "
        "auth checks) across routes without global state."
    ),
    "agent": (
        "An AI agent is a system that uses an LLM as a reasoning engine to decide "
        "which actions to take in pursuit of a goal. Unlike a simple prompt-response chain, "
        "agents can use tools, observe results, and iterate until the goal is satisfied. "
        "The key challenge is preventing infinite loops and managing context window limits."
    ),
    "chunk": (
        "Text chunking splits documents into smaller pieces before embedding. "
        "Chunk size is a trade-off: smaller chunks are more precise for retrieval "
        "but lose surrounding context; larger chunks preserve context but reduce precision. "
        "Recursive character splitters try natural boundaries (paragraphs, sentences) "
        "before falling back to character counts."
    ),
    "security": (
        "Enterprise AI security involves multiple layers: input validation to detect "
        "prompt injection attacks, PII masking before sending data to external LLMs, "
        "output validation to filter harmful content, and audit logging for compliance. "
        "Zero-trust principles apply: treat every LLM call as an external service call."
    ),
}

DEFAULT_RESPONSE = (
    "I understand your question. Based on the context provided, "
    "this is a well-structured query that touches on important concepts in "
    "enterprise AI systems. The answer depends on your specific requirements, "
    "but the key considerations are: system design, scalability, and maintainability. "
    "I recommend starting with the simplest approach and iterating based on real usage patterns."
)


class MockLLMProvider(BaseLLM):
    """
    Deterministic mock LLM provider.

    Returns realistic-shaped responses without making any network calls.
    Simulates token usage and latency so metrics code exercises the same paths.
    """

    def __init__(self, simulated_latency_ms: float = 50.0):
        """
        Args:
            simulated_latency_ms: Artificial delay to simulate real LLM latency.
                Set to 0 in tests for fast execution.
        """
        self.simulated_latency_ms = simulated_latency_ms
        self.model_name = "mock-llm-v1"
        logger.info("mock_llm_initialized", latency_ms=simulated_latency_ms)

    def _select_response(self, messages: List[ChatMessage]) -> str:
        """Pick a canned response based on keywords in the last user message."""
        # Find the last user message
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_msg = msg.content.lower()
                break

        # Check for keyword matches
        for keyword, response in CANNED_RESPONSES.items():
            if keyword in last_user_msg:
                return response

        return DEFAULT_RESPONSE

    async def complete(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Return a deterministic canned response after a simulated delay."""
        start = time.perf_counter()

        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000)

        content = self._select_response(messages)

        # Calculate token estimates for realistic usage reporting
        prompt_text = " ".join(m.content for m in messages)
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(content)

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        logger.info(
            "mock_llm_completion",
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

        return LLMResponse(
            content=content,
            usage=usage,
            model=self.model_name,
            latency_ms=latency_ms,
        )

    async def stream(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Simulate streaming by yielding words one at a time."""
        response = await self.complete(messages, max_tokens, temperature)
        words = response.content.split()
        for word in words:
            await asyncio.sleep(0.01)  # Simulate token-by-token streaming
            yield word + " "

    async def health_check(self) -> bool:
        """Mock is always healthy — it never makes network calls."""
        return True

    @property
    def provider_name(self) -> str:
        return "MockLLMProvider"
