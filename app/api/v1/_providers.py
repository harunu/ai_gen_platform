# WHY THIS FILE EXISTS:
# Centralized FastAPI dependency providers for AI layer objects.
# This is the wiring layer: reads config and returns the correct
# LLM provider (mock or real) and the vector store + embeddings.
# Route handlers import these as Depends() — never import LLM classes directly.

from typing import Tuple

from fastapi import Depends

from app.ai.embeddings.base import BaseEmbeddings
from app.ai.embeddings.provider import get_embeddings_provider
from app.ai.llm.base import BaseLLM
from app.core.config import Settings, get_settings
from app.repositories.chroma_store import get_chroma_store
from app.repositories.vector_store import VectorStore


def get_llm(settings: Settings = Depends(get_settings)) -> BaseLLM:
    """
    Return the configured LLM provider.

    - USE_MOCK_LLM=true (default): returns MockLLMProvider (no API calls, no cost)
    - USE_MOCK_LLM=false: returns OpenAICompatibleLLM (real API calls)

    This is the single place where the LLM implementation is selected.
    Swapping to a different provider (Anthropic, Azure) = change this function.
    """
    if settings.use_mock_llm:
        from app.ai.llm.mock_provider import MockLLMProvider
        return MockLLMProvider(simulated_latency_ms=50.0)
    else:
        from app.ai.llm.openai_compatible import OpenAICompatibleLLM
        return OpenAICompatibleLLM()


def get_vector_store_and_embeddings() -> Tuple[VectorStore, BaseEmbeddings]:
    """
    Return the vector store and embeddings provider.
    Both are singletons (lru_cache), so the model loads only once.
    """
    return get_chroma_store(), get_embeddings_provider()
