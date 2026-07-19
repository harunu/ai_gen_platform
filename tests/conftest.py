# WHY THIS FILE EXISTS:
# pytest discovers conftest.py automatically and shares its fixtures across all test files.
# Fixtures are reusable setup/teardown helpers injected into test functions by name.
# This file wires together: test HTTP client, mock LLM, in-memory vector store.

import asyncio
from typing import AsyncGenerator, List
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.ai.llm.base import BaseLLM, LLMResponse
from app.ai.llm.mock_provider import MockLLMProvider
from app.core.config import get_settings
from app.core.security import create_access_token
from app.main import create_app
from app.models.chat import ChatMessage, TokenUsage
from app.models.rag import DocumentMetadata, SourceCitation
from app.repositories.vector_store import VectorStore


# ── Event Loop ────────────────────────────────────────────────────────────────
# pytest-asyncio requires a session-scoped event loop for async fixtures.

@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── In-Memory Vector Store ────────────────────────────────────────────────────

class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store for tests.
    Does not require a running ChromaDB server.
    Uses brute-force cosine similarity for correctness.
    """

    def __init__(self):
        self._data = {}  # id → {embedding, document, metadata}

    async def upsert(self, ids, embeddings, documents, metadatas):
        for i, doc_id in enumerate(ids):
            self._data[doc_id] = {
                "embedding": embeddings[i],
                "document": documents[i],
                "metadata": metadatas[i],
            }

    async def query(self, query_embedding, top_k, where=None):
        import math

        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x**2 for x in a))
            mag_b = math.sqrt(sum(x**2 for x in b))
            if mag_a == 0 or mag_b == 0:
                return 0.0
            return dot / (mag_a * mag_b)

        results = []
        for doc_id, data in self._data.items():
            if where:
                meta = data["metadata"]
                # Simple equality filter
                skip = False
                for k, v in where.items():
                    if isinstance(v, dict) and "$in" in v:
                        if meta.get(k) not in v["$in"]:
                            skip = True
                            break
                    elif meta.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue

            score = cosine_sim(query_embedding, data["embedding"])
            results.append({
                "id": doc_id,
                "document": data["document"],
                "metadata": data["metadata"],
                "similarity_score": round(score, 4),
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    async def delete(self, ids):
        for doc_id in ids:
            self._data.pop(doc_id, None)

    async def count(self):
        return len(self._data)

    async def health_check(self):
        return True


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_vector_store() -> InMemoryVectorStore:
    """Fresh in-memory vector store for each test."""
    return InMemoryVectorStore()


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Mock LLM with zero latency for fast tests."""
    return MockLLMProvider(simulated_latency_ms=0)


@pytest.fixture
def sample_documents() -> List[dict]:
    """Sample documents for RAG pipeline tests."""
    return [
        {
            "content": (
                "Retrieval-Augmented Generation (RAG) combines document retrieval with LLM generation. "
                "It retrieves relevant chunks from a vector database and uses them as context for the LLM. "
                "RAG reduces hallucinations by grounding answers in retrieved facts."
            ),
            "metadata": DocumentMetadata(
                source="rag_guide.md",
                title="RAG Guide",
                tags=["rag", "ai"],
            ),
        },
        {
            "content": (
                "FastAPI is a modern Python web framework for building APIs. "
                "It uses type hints for automatic validation and documentation. "
                "FastAPI is built on Starlette and supports async endpoints natively."
            ),
            "metadata": DocumentMetadata(
                source="fastapi_guide.md",
                title="FastAPI Guide",
                tags=["fastapi", "python"],
            ),
        },
        {
            "content": (
                "ChromaDB is an open-source vector database for AI applications. "
                "It supports persistent storage, metadata filtering, and cosine similarity search. "
                "ChromaDB can run locally or as a server in Docker."
            ),
            "metadata": DocumentMetadata(
                source="chroma_guide.md",
                title="ChromaDB Guide",
                tags=["chromadb", "vector-db"],
            ),
        },
    ]


@pytest_asyncio.fixture
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP test client for the FastAPI app.

    ASGITransport allows httpx to call the FastAPI app directly
    without starting a real server. Tests run in-process — fast and reliable.
    """
    app = create_app()

    # Override the LLM provider with mock (no API calls in tests)
    from app.api.v1 import _providers
    original_get_llm = _providers.get_llm

    def mock_get_llm(settings=None):
        return MockLLMProvider(simulated_latency_ms=0)

    _providers.get_llm = mock_get_llm

    # Override vector store with in-memory store (no ChromaDB needed)
    original_get_vs = _providers.get_vector_store_and_embeddings

    def mock_get_vs():
        from tests.conftest import InMemoryVectorStore
        from app.ai.embeddings.provider import get_embeddings_provider
        return InMemoryVectorStore(), get_embeddings_provider()

    _providers.get_vector_store_and_embeddings = mock_get_vs

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    # Restore original providers
    _providers.get_llm = original_get_llm
    _providers.get_vector_store_and_embeddings = original_get_vs


@pytest.fixture
def auth_headers() -> dict:
    """JWT Authorization headers for a test user."""
    token = create_access_token(subject="test_user", extra_claims={"role": "admin"})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api_key_headers() -> dict:
    """API key headers for protected endpoints."""
    settings = get_settings()
    return {"X-API-Key": settings.api_key}
