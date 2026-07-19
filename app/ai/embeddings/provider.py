# WHY THIS FILE EXISTS:
# Sentence-transformers implementation of the embeddings interface.
# sentence-transformers is a free, local library from HuggingFace that runs
# transformer models entirely on your machine — no API key, no cost, no network.
# Model: all-MiniLM-L6-v2 produces 384-dimensional vectors, is fast, and
# works well for English semantic similarity.

import asyncio
from functools import lru_cache
from typing import List

from app.ai.embeddings.base import BaseEmbeddings
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """
    Embeddings provider using HuggingFace sentence-transformers.

    The model is loaded once at startup and kept in memory.
    encode() is synchronous (CPU-bound), so we run it in a thread pool
    via asyncio.get_event_loop().run_in_executor() to avoid blocking
    FastAPI's async event loop.

    Why not just call it synchronously?
    FastAPI runs on an async event loop. A slow synchronous call (like model
    inference) would block ALL other requests until it finishes. Running it
    in a thread pool lets other requests proceed concurrently.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._model = None   # Lazy-loaded on first use
        self._dimension: int = 0

    def _load_model(self) -> None:
        """Load the model on first use (lazy initialization)."""
        if self._model is not None:
            return

        # Import here so the module loads even if torch isn't installed
        # (e.g. in environments where you only use the mock)
        from sentence_transformers import SentenceTransformer

        logger.info("loading_embedding_model", model=self._model_name)
        self._model = SentenceTransformer(self._model_name)

        # Determine vector dimension by embedding a test string
        test_embedding = self._model.encode(["test"])
        self._dimension = len(test_embedding[0])
        logger.info("embedding_model_loaded", model=self._model_name, dimension=self._dimension)

    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous encoding — runs in a thread pool."""
        self._load_model()
        # show_progress_bar=False: suppress tqdm output in production logs
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize so cosine sim = dot product
        )
        return [emb.tolist() for emb in embeddings]

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts asynchronously by offloading to a thread pool.

        asyncio.get_event_loop().run_in_executor(None, func) runs `func`
        in the default ThreadPoolExecutor without blocking the event loop.
        """
        if not texts:
            return []

        loop = asyncio.get_event_loop()
        # run_in_executor takes a callable — use a lambda to pass arguments
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._encode_sync(texts),
        )

        logger.info("texts_embedded", count=len(texts), model=self._model_name)
        return embeddings

    def get_dimension(self) -> int:
        """Return vector dimension, loading the model if needed."""
        if self._dimension == 0:
            self._load_model()
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name


@lru_cache(maxsize=1)
def get_embeddings_provider() -> SentenceTransformerEmbeddings:
    """
    Return the singleton embeddings provider.
    lru_cache ensures the model is only loaded once per process.
    """
    return SentenceTransformerEmbeddings()
