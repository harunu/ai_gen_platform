#!/usr/bin/env python3
# WHY THIS FILE EXISTS:
# Wipes the ChromaDB collection and rebuilds the vector index from scratch.
# Use this when:
# - You changed the embedding model (vectors are no longer comparable)
# - You changed chunk size/overlap significantly
# - You want to start fresh after adding/removing documents
# Run via: make rebuild-index

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import setup_logging
setup_logging("INFO")

from app.core.logging import get_logger
from app.ai.embeddings.provider import get_embeddings_provider
from app.repositories.chroma_store import get_chroma_store
from app.repositories.document_repository import DocumentRepository
from app.services.ingestion_service import IngestionService

logger = get_logger(__name__)


async def main() -> None:
    vector_store = get_chroma_store()

    if not await vector_store.health_check():
        logger.error("chroma_unreachable")
        sys.exit(1)

    # Step 1: Get current count
    before_count = await vector_store.count()
    logger.info("rebuild_start", vectors_before=before_count)
    print(f"  Current vectors in index: {before_count}")
    print("  Deleting collection...")

    # Step 2: Wipe the collection
    await vector_store.delete_all()

    # Step 3: Re-ingest from sample docs
    data_dir = Path(__file__).parent.parent / "data" / "sample_docs"
    embeddings = get_embeddings_provider()
    repo = DocumentRepository(vector_store=vector_store, embeddings=embeddings)
    service = IngestionService(document_repository=repo)

    results = await service.ingest_directory(str(data_dir))
    total_chunks = sum(c for _, c in results)

    after_count = await vector_store.count()
    logger.info("rebuild_complete", documents=len(results), chunks=total_chunks, vectors_after=after_count)
    print(f"✓ Rebuilt index: {len(results)} documents, {total_chunks} chunks, {after_count} vectors")


if __name__ == "__main__":
    asyncio.run(main())
