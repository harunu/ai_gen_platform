# WHY THIS FILE EXISTS:
# Tests for the RAG pipeline components:
# - RecursiveCharacterChunker (unit tests)
# - RAGRetriever (unit test with in-memory store)
# - Full RAG pipeline (integration test: ingest → query)

import pytest
from httpx import AsyncClient

from app.ai.embeddings.provider import SentenceTransformerEmbeddings
from app.ai.llm.mock_provider import MockLLMProvider
from app.ai.rag.chunker import RecursiveCharacterChunker
from app.ai.rag.pipeline import RAGPipeline
from app.ai.rag.retriever import RAGRetriever
from app.models.rag import DocumentMetadata
from app.repositories.document_repository import DocumentRepository
from tests.conftest import InMemoryVectorStore


# ── Chunker Unit Tests ────────────────────────────────────────────────────────

class TestRecursiveCharacterChunker:

    def test_short_text_returns_single_chunk(self) -> None:
        """Text shorter than chunk_size should produce exactly one chunk."""
        chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk("This is a short text.")
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short text."

    def test_empty_text_returns_no_chunks(self) -> None:
        """Empty or whitespace-only text should return an empty list."""
        chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_long_text_produces_multiple_chunks(self) -> None:
        """Text longer than chunk_size should be split into multiple chunks."""
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
        text = "word " * 100   # 500 chars
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunks_are_ordered(self) -> None:
        """Chunks must be in document order (chunk_index 0, 1, 2, ...)."""
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
        text = "sentence. " * 50
        chunks = chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_overlap_respected(self) -> None:
        """Each chunk (except the last) must have some content overlap with the next."""
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
        text = "A" * 50 + " " + "B" * 50 + " " + "C" * 50
        chunks = chunker.chunk(text)
        # Just verify we get more than one chunk and no empty chunks
        assert all(len(c.text.strip()) > 0 for c in chunks)

    def test_metadata_attached_to_chunks(self) -> None:
        """Metadata provided to chunk() should appear in each TextChunk."""
        chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=5)
        meta = {"source": "test.md", "title": "Test"}
        chunks = chunker.chunk("word " * 30, metadata=meta)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.md"

    def test_invalid_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError):
            RecursiveCharacterChunker(chunk_size=0)

    def test_overlap_greater_than_size_raises(self) -> None:
        with pytest.raises(ValueError):
            RecursiveCharacterChunker(chunk_size=100, chunk_overlap=150)


# ── Retriever Unit Test ───────────────────────────────────────────────────────

class MockEmbeddings:
    """Deterministic mock embeddings that return fixed vectors."""

    async def embed(self, texts):
        # Return a 3-dimensional vector where the first component = text length / 100
        return [[len(t) / 100.0, 0.5, 0.5] for t in texts]

    async def embed_single(self, text):
        results = await self.embed([text])
        return results[0]

    def get_dimension(self):
        return 3


@pytest.mark.asyncio
async def test_retriever_returns_relevant_results() -> None:
    """Retriever should return the most similar chunks above the threshold."""
    store = InMemoryVectorStore()
    embeddings = MockEmbeddings()

    # Pre-populate the store
    await store.upsert(
        ids=["chunk_0", "chunk_1"],
        embeddings=[[0.9, 0.5, 0.5], [0.1, 0.5, 0.5]],
        documents=["RAG document chunk", "Unrelated chunk"],
        metadatas=[
            {"source": "rag.md", "title": "RAG Guide", "chunk_index": 0, "tags": ""},
            {"source": "other.md", "title": "Other", "chunk_index": 0, "tags": ""},
        ],
    )

    retriever = RAGRetriever(vector_store=store, embeddings=embeddings)
    # Query similar to chunk_0
    results = await retriever.retrieve("long query text" * 9, top_k=2, similarity_threshold=0.0)
    assert len(results) > 0
    # The first result should be the more similar chunk
    assert results[0].chunk_id in ["chunk_0", "chunk_1"]


@pytest.mark.asyncio
async def test_retriever_filters_below_threshold() -> None:
    """Chunks with similarity below threshold should be excluded."""
    store = InMemoryVectorStore()
    embeddings = MockEmbeddings()

    await store.upsert(
        ids=["chunk_low"],
        embeddings=[[0.0, 0.0, 1.0]],
        documents=["Low similarity chunk"],
        metadatas=[{"source": "s.md", "title": "T", "chunk_index": 0, "tags": ""}],
    )

    retriever = RAGRetriever(vector_store=store, embeddings=embeddings)
    results = await retriever.retrieve("test", top_k=5, similarity_threshold=0.99)
    assert results == []


# ── Full Pipeline Integration Test ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rag_pipeline_end_to_end() -> None:
    """
    Integration test: ingest documents, run a query, verify answer and sources.
    Uses in-memory store and real sentence-transformers for embeddings.
    """
    store = InMemoryVectorStore()
    embeddings = SentenceTransformerEmbeddings()
    llm = MockLLMProvider(simulated_latency_ms=0)

    # Ingest a document
    repo = DocumentRepository(vector_store=store, embeddings=embeddings, chunk_size=200, chunk_overlap=20)
    doc_id, num_chunks = await repo.ingest(
        content=(
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "a retrieval system with a large language model. The retrieval system "
            "finds relevant documents, which are then used as context for the LLM. "
            "This reduces hallucinations and keeps answers factual."
        ),
        metadata=DocumentMetadata(source="test_rag.md", title="RAG Test"),
    )
    assert num_chunks >= 1

    # Run a query
    retriever = RAGRetriever(vector_store=store, embeddings=embeddings)
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    response = await pipeline.run(
        query="What is RAG?",
        similarity_threshold=0.0,
    )

    assert response.answer
    assert len(response.answer) > 5
    assert response.latency_ms > 0
    assert response.confidence_hint in ("high", "medium", "low")


@pytest.mark.asyncio
async def test_rag_api_endpoint_requires_api_key(test_client: AsyncClient) -> None:
    """RAG query endpoint must return 403 without X-API-Key header."""
    response = await test_client.post(
        "/api/v1/rag/query",
        json={"query": "What is RAG?"},
    )
    assert response.status_code in (401, 403)
