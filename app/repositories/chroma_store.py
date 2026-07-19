# WHY THIS FILE EXISTS:
# ChromaDB implementation of the VectorStore interface.
# ChromaDB is an open-source vector database that runs locally in Docker.
# It supports cosine similarity search, metadata filtering, and persistent storage.
# This class is the ONLY place in the codebase that imports chromadb.

from functools import lru_cache
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.repositories.vector_store import VectorStore

logger = get_logger(__name__)


class ChromaVectorStore(VectorStore):
    """
    ChromaDB vector store implementation.

    ChromaDB concepts:
    - Collection: Like a table — holds vectors, documents, and metadata.
      All vectors in a collection must have the same dimension.
    - Embedding function: ChromaDB can embed for you, but we embed externally
      (in our embeddings provider) and pass pre-computed vectors. This gives us
      full control over the embedding model and batching.
    - Distance function: We use cosine similarity (higher = more similar).
      ChromaDB also supports L2 (Euclidean) and inner product.

    ChromaDB HTTP client vs embedded:
    - Embedded: ChromaDB runs in the same process as our app. Simple but can't
      scale horizontally.
    - HTTP client (used here): ChromaDB runs as a separate service (in Docker).
      Our app connects to it via HTTP. Supports multiple app instances.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._collection_name = settings.chroma_collection_name

        # Connect to the ChromaDB HTTP server running in Docker
        self._client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = None
        logger.info(
            "chroma_client_initialized",
            host=settings.chroma_host,
            port=settings.chroma_port,
            collection=self._collection_name,
        )

    def _get_collection(self):
        """
        Lazily get or create the ChromaDB collection.

        get_or_create_collection: Creates the collection if it doesn't exist,
        returns existing one if it does. This is idempotent — safe to call on
        every request.

        metadata={"hnsw:space": "cosine"}: Use cosine similarity for distance.
        HNSW (Hierarchical Navigable Small World) is the index algorithm —
        an approximate nearest neighbor search method that's fast and memory-efficient.
        """
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Insert or update vectors. ChromaDB's upsert is idempotent:
        if an ID already exists, it overwrites; otherwise it inserts.
        This makes re-ingestion of the same document safe.
        """
        try:
            collection = self._get_collection()
            # ChromaDB upsert is synchronous. In our async context, this is
            # acceptable because ingestion is typically a background/script operation.
            # For high-throughput ingestion, use run_in_executor.
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("chroma_upsert", count=len(ids))
        except Exception as e:
            logger.error("chroma_upsert_failed", error=str(e), count=len(ids))
            raise VectorStoreError(f"ChromaDB upsert failed: {e}")

    async def query(
        self,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity search.

        ChromaDB returns results sorted by distance (lowest first for cosine = highest similarity).
        We convert distance to similarity: similarity = 1 - distance.
        """
        try:
            collection = self._get_collection()

            query_params: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": min(top_k, max(1, collection.count())),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_params["where"] = where

            results = collection.query(**query_params)

            # Reshape ChromaDB's batch response format into a flat list of dicts
            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            output = []
            for i, doc_id in enumerate(ids):
                # Convert distance → similarity score (cosine distance = 1 - cosine similarity)
                similarity = max(0.0, 1.0 - distances[i])
                output.append({
                    "id": doc_id,
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "similarity_score": round(similarity, 4),
                })

            return output

        except Exception as e:
            logger.error("chroma_query_failed", error=str(e))
            raise VectorStoreError(f"ChromaDB query failed: {e}")

    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID."""
        try:
            collection = self._get_collection()
            collection.delete(ids=ids)
            logger.info("chroma_delete", count=len(ids))
        except Exception as e:
            raise VectorStoreError(f"ChromaDB delete failed: {e}")

    async def count(self) -> int:
        """Return the total number of vectors in the collection."""
        try:
            return self._get_collection().count()
        except Exception as e:
            raise VectorStoreError(f"ChromaDB count failed: {e}")

    async def health_check(self) -> bool:
        """Check if ChromaDB is reachable."""
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False

    async def delete_all(self) -> None:
        """Delete the entire collection and recreate it (efficient wipe)."""
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = None  # Force recreation on next use
            logger.info("chroma_collection_deleted", collection=self._collection_name)
        except Exception as e:
            raise VectorStoreError(f"ChromaDB delete_all failed: {e}")


@lru_cache(maxsize=1)
def get_chroma_store() -> ChromaVectorStore:
    """Return the singleton ChromaDB store."""
    return ChromaVectorStore()
