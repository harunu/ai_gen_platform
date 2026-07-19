# WHY THIS FILE EXISTS:
# Business logic for the chat endpoint.
# Routes keep HTTP concerns (request parsing, response formatting).
# Services keep business logic (calling LLM, building messages, logging).
# This separation makes it easy to test business logic without HTTP.

import time

from app.ai.guardrails.injection import check_prompt_injection
from app.ai.guardrails.output_validator import validate_llm_output
from app.ai.guardrails.pii import check_and_mask_pii
from app.ai.llm.base import BaseLLM
from app.ai.telemetry.cost_tracker import track_llm_call
from app.core.logging import get_logger
from app.models.chat import ChatMessage, ChatRequest, ChatResponse

logger = get_logger(__name__)


class ChatService:
    """Handles chat completions through the LLM."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    async def chat(self, request: ChatRequest, user: str) -> ChatResponse:
        """
        Process a chat request.

        Applies guardrails to the last user message, calls the LLM,
        validates output, and returns the response.
        """
        start = time.perf_counter()

        # Apply guardrails to the last user message
        # (don't modify assistant history — that would break conversation continuity)
        messages = list(request.messages)
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                last_user_idx = i
                break

        if last_user_idx is not None:
            original = messages[last_user_idx].content
            check_prompt_injection(original, context="chat_input")
            cleaned = check_and_mask_pii(original)
            if cleaned != original:
                messages[last_user_idx] = ChatMessage(role="user", content=cleaned)

        # Call LLM
        llm_response = await self.llm.complete(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Validate output
        validated_content = validate_llm_output(llm_response.content, context="chat_response")

        # Track cost
        track_llm_call(
            model=llm_response.model,
            prompt_tokens=llm_response.usage.prompt_tokens,
            completion_tokens=llm_response.usage.completion_tokens,
            latency_ms=llm_response.latency_ms,
        )

        total_latency = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "chat_complete",
            user=user,
            model=llm_response.model,
            total_tokens=llm_response.usage.total_tokens,
            latency_ms=total_latency,
        )

        return ChatResponse(
            message=ChatMessage(role="assistant", content=validated_content),
            model=llm_response.model,
            usage=llm_response.usage,
            latency_ms=total_latency,
        )
