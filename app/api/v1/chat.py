# WHY THIS FILE EXISTS:
# Chat endpoint: JWT-protected conversational AI.
# Routes are thin: they parse HTTP, call services, return responses.
# All business logic lives in ChatService.

from fastapi import APIRouter, Depends, Request

from app.ai.llm.base import BaseLLM
from app.api.v1._providers import get_llm
from app.core.dependencies import CurrentUser
from app.core.rate_limit import AI_RATE_LIMIT, limiter
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, summary="Chat with the AI")
@limiter.limit(AI_RATE_LIMIT)
async def chat(
    request: Request,
    body: ChatRequest,
    current_user: CurrentUser,             # Validates JWT; provides username
    llm: BaseLLM = Depends(get_llm),       # Injected LLM (mock or real)
) -> ChatResponse:
    """
    Send a conversation and receive an AI response.

    Requires a valid JWT token in the Authorization header.
    Apply guardrails (injection detection, PII masking) before calling the LLM.

    Example request:
    ```json
    {
        "messages": [
            {"role": "user", "content": "What is RAG?"}
        ]
    }
    ```
    """
    service = ChatService(llm=llm)
    return await service.chat(request=body, user=current_user)
