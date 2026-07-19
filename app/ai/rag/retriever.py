# WHY THIS FILE EXISTS:
# Handles semantic similarity search against the vector store.
# Takes a query, embeds it, searches ChromaDB, and applies filters.
# This is the "R" in RAG — the retrieval step.

import time
from typing import List, Optional

from app.ai.embeddings.base import BaseEmbeddings
from app.ai.embeddings.provider import get_embeddings_provider
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.rag import SourceCitation
from app.repositories.vector_store import VectorStore

logger = get_logger(__name__)


class RAGRetriever:
    """
    Retrieves relevant document chunks for a given query.

    Flow:
    1. Embed the query using the same model used during ingestion
       (CRITICAL: query and documents must use the same embedding model)
    2. Perform cosine similarity search in ChromaDB
    3. Apply metadata filters if specified
    4. Filter out chunks below the similarity threshold
    5. Return as SourceCitation objects
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embeddings: Optional[BaseEmbeddings] = None,
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings or get_embeddings_provider()
        self.settings = get_settings()

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_tags: Optional[List[str]] = None,
    ) -> List[SourceCitation]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: The user's question (will be embedded).
            top_k: Number of results to return (default from settings).
            similarity_threshold: Minimum similarity score (0–1) to include.
            filter_tags: Only retrieve chunks with matching tags in metadata.

        Returns:
            List of SourceCitation objects sorted by similarity (highest first).
        """
        start = time.perf_counter()

        k = top_k or self.settings.top_k
        threshold = similarity_threshold if similarity_threshold is not None else self.settings.similarity_threshold

        # Step 1: Embed the query
        # The query embedding must use the EXACT same model as the document embeddings.
        # If they differ, similarity scores are meaningless garbage.
        query_embedding = await self.embeddings.embed_single(query)

        # Step 2: Build metadata filter
        # ChromaDB supports filtering by metadata fields using a dict of conditions.
        where_filter = None
        if filter_tags:
            # Filter for documents whose 'tags' metadata contains any of the filter_tags
            where_filter = {"tags": {"$in": filter_tags}}

        # Step 3: Query the vector store
        results = await self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k,
            where=where_filter,
        )

        # Step 4: Convert raw results to SourceCitation objects and apply threshold
        citations: List[SourceCitation] = []
        for result in results:
            # ChromaDB returns distances (lower = more similar for L2) or
            # similarity scores (higher = more similar for cosine).
            # Our ChromaDB store returns cosine similarity scores (0–1).
            score = result.get("similarity_score", 0.0)

            if score < threshold:
                logger.debug(
                    "chunk_below_threshold",
                    chunk_id=result.get("id", ""),
                    score=score,
                    threshold=threshold,
                )
                continue

            metadata = result.get("metadata", {})
            citations.append(SourceCitation(
                chunk_id=result.get("id", ""),
                content=result.get("document", ""),
                source=metadata.get("source", "unknown"),
                title=metadata.get("title"),
                similarity_score=score,
                chunk_index=metadata.get("chunk_index", 0),
            ))

        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "retrieval_complete",
            query_preview=query[:80],
            top_k=k,
            threshold=threshold,
            results_found=len(citations),
            latency_ms=elapsed,
        )

        return citations
