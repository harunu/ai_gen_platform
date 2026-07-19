# WHY THIS FILE EXISTS:
# An async OpenAI-compatible LLM client using httpx.
# "OpenAI-compatible" means this works with OpenAI, Azure OpenAI, Ollama,
# vLLM, LM Studio, or any provider that implements the /v1/chat/completions API.
# Using httpx directly (rather than the openai SDK) keeps dependencies minimal
# and makes the retry/timeout logic fully visible and controllable.

import time
from typing import AsyncIterator, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ai.llm.base import BaseLLM, LLMResponse
from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.models.chat import ChatMessage, TokenUsage

logger = get_logger(__name__)


class OpenAICompatibleLLM(BaseLLM):
    """
    Async LLM client for any OpenAI-compatible API.

    Uses httpx.AsyncClient with:
    - Configurable timeout
    - Automatic retry with exponential backoff (via tenacity)
    - Full structured logging of every request
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_seconds
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature

        # httpx.AsyncClient is a connection pool — create once, reuse across requests.
        # This is more efficient than creating a new client per request.
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self.timeout),
        )
        logger.info("openai_compatible_llm_initialized", model=self.model, base_url=self.base_url)

    def _build_messages_payload(self, messages: List[ChatMessage]) -> list:
        """Convert internal ChatMessage objects to the OpenAI API format."""
        return [{"role": m.role, "content": m.content} for m in messages]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Call the /v1/chat/completions endpoint with automatic retry.

        tenacity's @retry decorator:
        - Retries up to 3 times on any exception
        - Waits 1s, 2s, 4s between attempts (exponential backoff)
        - reraise=True: re-raises the last exception if all retries fail
        """
        start = time.perf_counter()

        payload = {
            "model": self.model,
            "messages": self._build_messages_payload(messages),
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()  # Raises httpx.HTTPStatusError for 4xx/5xx

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage_data = data.get("usage", {})

            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

            logger.info(
                "llm_completion",
                model=self.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                latency_ms=latency_ms,
            )

            return LLMResponse(
                content=content,
                usage=usage,
                model=self.model,
                latency_ms=latency_ms,
            )

        except httpx.HTTPStatusError as e:
            logger.error("llm_http_error", status_code=e.response.status_code, error=str(e))
            raise LLMError(f"LLM API returned HTTP {e.response.status_code}")
        except httpx.TimeoutException:
            logger.error("llm_timeout", timeout_seconds=self.timeout)
            raise LLMError(f"LLM API timed out after {self.timeout}s")
        except Exception as e:
            logger.error("llm_unexpected_error", error=str(e))
            raise LLMError(f"Unexpected LLM error: {e}")

    async def stream(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens using Server-Sent Events (SSE).

        OpenAI's streaming API sends data: {"choices": [{"delta": {"content": "..."}}]}
        lines terminated by data: [DONE].
        """
        payload = {
            "model": self.model,
            "messages": self._build_messages_payload(messages),
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": True,
        }

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]   # Strip "data: " prefix
                    if data_str.strip() == "[DONE]":
                        break
                    import json
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"]
                        if token := delta.get("content"):
                            yield token
                    except (KeyError, json.JSONDecodeError):
                        continue
        except httpx.HTTPStatusError as e:
            raise LLMError(f"LLM stream error: HTTP {e.response.status_code}")

    async def health_check(self) -> bool:
        """Ping the API with a minimal request to verify connectivity."""
        try:
            response = await self._client.get(f"{self.base_url}/models", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying HTTP connection pool. Call on app shutdown."""
        await self._client.aclose()

    @property
    def provider_name(self) -> str:
        return f"OpenAICompatible({self.model})"
