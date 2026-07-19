# WHY THIS FILE EXISTS:
# Abstract interface for the vector store.
# By depending on this interface (not ChromaDB directly), higher layers
# are completely decoupled from the storage backend.
# Swapping ChromaDB for Pinecone, Weaviate, or Azure AI Search means
# writing one new class that implements this interface — nothing else changes.

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStore(ABC):
    """
    Abstract vector store interface.

    All vector databases support the same fundamental operations:
    - upsert: insert or update vectors with metadata
    - query: nearest-neighbor search by vector similarity
    - delete: remove vectors by ID
    - count: number of stored vectors
    """

    @abstractmethod
    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Insert or update vectors in the store.

        Args:
            ids: Unique string IDs for each vector.
            embeddings: Float vectors (must match the store's expected dimension).
            documents: Raw text corresponding to each vector (stored alongside).
            metadatas: Arbitrary key-value metadata for each vector.
        """
        ...

    @abstractmethod
    async def query(
        self,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar vectors.

        Args:
            query_embedding: The query vector to search against.
            top_k: Number of results to return.
            where: Optional metadata filter (format depends on backend).

        Returns:
            List of dicts with keys: id, document, metadata, similarity_score.
        """
        ...

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by their IDs."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of vectors in the store."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the store is reachable and operational."""
        ...

    async def delete_all(self) -> None:
        """
        Delete all vectors. Default: get all IDs and delete them.
        Override in subclasses for a more efficient bulk-delete operation.
        """
        raise NotImplementedError("delete_all not implemented for this backend")
