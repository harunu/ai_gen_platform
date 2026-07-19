# WHY THIS FILE EXISTS:
# Abstract interface for embedding providers.
# Swapping embedding models (e.g. from sentence-transformers to OpenAI ada-002)
# requires only a new class implementing this interface — zero changes elsewhere.

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddings(ABC):
    """Abstract base class for text embedding providers."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts into vectors.

        Args:
            texts: List of strings to embed. Batching is handled internally.

        Returns:
            List of float vectors, one per input text.
            All vectors have the same dimensionality.
        """
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the vector dimensionality for this embedding model."""
        ...

    async def embed_single(self, text: str) -> List[float]:
        """Convenience method: embed a single string."""
        results = await self.embed([text])
        return results[0]

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        return self.__class__.__name__
