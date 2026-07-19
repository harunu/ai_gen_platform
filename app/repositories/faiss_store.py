# WHY THIS FILE EXISTS:
# FAISS (Facebook AI Similarity Search) adapter — documented as an optional
# alternative to ChromaDB.
#
# When to use FAISS instead of ChromaDB:
# - You need maximum query throughput (FAISS is faster for pure similarity search)
# - You want to run entirely in-process without a separate server
# - You're fine managing index persistence manually
#
# When to stick with ChromaDB:
# - You want metadata filtering alongside similarity search
# - You want a persistent server with an HTTP API
# - You don't want to manage index serialization
#
# NOTE: faiss-cpu is NOT in requirements.txt. Add it to use this adapter:
#   pip install faiss-cpu
#
# This file is documented but not used by default.
# To activate: replace ChromaVectorStore with FAISSVectorStore in dependencies.py.

from typing import Any, Dict, List, Optional

from app.core.logging import get_logger
from app.repositories.vector_store import VectorStore

logger = get_logger(__name__)


class FAISSVectorStore(VectorStore):
    """
    FAISS in-process vector store adapter.

    Architecture note:
    FAISS stores vectors in a flat index (brute-force exact search) or
    an IVF index (approximate search with Voronoi partitions).
    For < 100k vectors, flat index is fine. Above that, use IVFFlat.

    FAISS doesn't store document text or metadata — you need a separate
    store (dict or SQLite) to map vector IDs back to content.
    """

    def __init__(self, dimension: int = 384) -> None:
        """
        Args:
            dimension: Vector dimension (must match your embedding model).
                       all-MiniLM-L6-v2 produces 384-dimensional vectors.
        """
        try:
            import faiss
            import numpy as np
            self._faiss = faiss
            self._np = np
        except ImportError:
            raise ImportError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            )

        self.dimension = dimension
        # IndexFlatIP: flat index with inner product (equivalent to cosine if normalized)
        self._index = faiss.IndexFlatIP(dimension)
        # ID mapping: FAISS uses integer IDs internally; we map to string IDs
        self._id_to_str: Dict[int, str] = {}
        self._str_to_int: Dict[str, int] = {}
        # Document and metadata storage alongside the index
        self._documents: Dict[str, str] = {}
        self._metadatas: Dict[str, Dict[str, Any]] = {}
        self._next_id = 0
        logger.info("faiss_store_initialized", dimension=dimension)

    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        vectors = self._np.array(embeddings, dtype=self._np.float32)
        for i, str_id in enumerate(ids):
            if str_id in self._str_to_int:
                # Update metadata and document (FAISS doesn't support in-place update)
                self._documents[str_id] = documents[i]
                self._metadatas[str_id] = metadatas[i]
            else:
                int_id = self._next_id
                self._next_id += 1
                self._id_to_str[int_id] = str_id
                self._str_to_int[str_id] = int_id
                self._index.add(vectors[i:i+1])
                self._documents[str_id] = documents[i]
                self._metadatas[str_id] = metadatas[i]
        logger.info("faiss_upsert", count=len(ids))

    async def query(
        self,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._index.ntotal == 0:
            return []

        vector = self._np.array([query_embedding], dtype=self._np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            str_id = self._id_to_str.get(int(idx))
            if str_id is None:
                continue

            # FAISS doesn't support metadata filtering — apply post-query if needed
            if where:
                meta = self._metadatas.get(str_id, {})
                match = all(meta.get(k) == v for k, v in where.items())
                if not match:
                    continue

            results.append({
                "id": str_id,
                "document": self._documents.get(str_id, ""),
                "metadata": self._metadatas.get(str_id, {}),
                "similarity_score": round(float(score), 4),
            })
        return results

    async def delete(self, ids: List[str]) -> None:
        # FAISS IndexFlatIP doesn't support element removal.
        # For production, use faiss.IndexIDMap or rebuild the index.
        logger.warning("faiss_delete_not_supported_rebuild_index")

    async def count(self) -> int:
        return self._index.ntotal

    async def health_check(self) -> bool:
        return True  # Always healthy — no external service
