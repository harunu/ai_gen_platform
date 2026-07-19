# WHY THIS FILE EXISTS:
# Abstracts document CRUD operations: chunking, embedding, and storing
# documents in the vector store. This is the "write path" for RAG.
# The read path is in ai/rag/retriever.py.

from typing import List, Optional

from app.ai.embeddings.base import BaseEmbeddings
from app.ai.rag.chunker import RecursiveCharacterChunker, TextChunk
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.rag import DocumentMetadata
from app.repositories.vector_store import VectorStore
from app.utils.ids import generate_document_id, generate_id
from app.utils.text import clean_text

logger = get_logger(__name__)


class DocumentRepository:
    """
    Handles the write path for documents:
    clean → chunk → embed → store.

    Separating this from the retriever enforces CQRS-like separation:
    - DocumentRepository  = write side (ingestion)
    - RAGRetriever        = read side (querying)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embeddings: BaseEmbeddings,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        settings = get_settings()
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
        )

    async def ingest(
        self,
        content: str,
        metadata: DocumentMetadata,
    ) -> tuple[str, int]:
        """
        Full ingestion pipeline for a single document.

        Returns:
            (document_id, num_chunks_created)
        """
        # 1. Clean the text
        cleaned = clean_text(content)
        if not cleaned:
            raise ValueError("Document content is empty after cleaning")

        # 2. Generate a stable document ID from the source
        doc_id = generate_document_id(metadata.source)

        # 3. Chunk the document
        chunk_metadata = {
            "source": metadata.source,
            "title": metadata.title or metadata.source,
            "author": metadata.author or "",
            "tags": ",".join(metadata.tags),  # ChromaDB metadata values must be strings
            "document_id": doc_id,
        }
        chunks: List[TextChunk] = self.chunker.chunk(cleaned, metadata=chunk_metadata)

        if not chunks:
            raise ValueError("No chunks produced from document")

        # 4. Embed all chunks in one batch (more efficient than one-by-one)
        texts = [chunk.text for chunk in chunks]
        embeddings = await self.embeddings.embed(texts)

        # 5. Prepare ChromaDB upsert data
        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            ids.append(chunk_id)
            metadatas.append({
                **chunk.metadata,
                "chunk_index": i,
                "char_count": chunk.char_count,
            })

        # 6. Store in the vector store
        await self.vector_store.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(
            "document_ingested",
            document_id=doc_id,
            source=metadata.source,
            num_chunks=len(chunks),
            total_chars=len(cleaned),
        )

        return doc_id, len(chunks)

    async def ingest_from_file(
        self,
        file_path: str,
        metadata: DocumentMetadata,
    ) -> tuple[str, int]:
        """Read a file from disk and ingest it."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return await self.ingest(content, metadata)
