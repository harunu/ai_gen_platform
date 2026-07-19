#!/usr/bin/env python3
# WHY THIS FILE EXISTS:
# Standalone script that ingests all documents from data/sample_docs/ into ChromaDB.
# Run via: docker compose run --rm app python scripts/ingest_documents.py
# Or:      make ingest

import asyncio
import sys
from pathlib import Path

# Add the project root to sys.path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.embeddings.provider import get_embeddings_provider
from app.core.logging import setup_logging
from app.repositories.chroma_store import get_chroma_store
from app.repositories.document_repository import DocumentRepository
from app.services.ingestion_service import IngestionService

# Initialize logging before anything else
setup_logging("INFO")

from app.core.logging import get_logger
logger = get_logger(__name__)


async def main() -> None:
    """
    Ingest all documents from data/sample_docs/.

    This script demonstrates the full ingestion pipeline:
    1. Connect to ChromaDB
    2. Load the embedding model
    3. Read each document from disk
    4. Chunk, embed, and store in ChromaDB
    """
    # Resolve the data directory relative to the project root
    data_dir = Path(__file__).parent.parent / "data" / "sample_docs"

    if not data_dir.exists():
        logger.error("data_dir_not_found", path=str(data_dir))
        sys.exit(1)

    logger.info("ingestion_script_start", data_dir=str(data_dir))

    # Wire up dependencies
    vector_store = get_chroma_store()
    embeddings = get_embeddings_provider()
    repo = DocumentRepository(vector_store=vector_store, embeddings=embeddings)
    service = IngestionService(document_repository=repo)

    # Check ChromaDB connectivity before starting
    if not await vector_store.health_check():
        logger.error("chroma_unreachable", hint="Is ChromaDB running? Try: docker compose up chroma")
        sys.exit(1)

    # Run ingestion
    results = await service.ingest_directory(str(data_dir))

    total_chunks = sum(chunks for _, chunks in results)
    logger.info(
        "ingestion_script_complete",
        documents_ingested=len(results),
        total_chunks=total_chunks,
    )
    print(f"\n✓ Ingested {len(results)} documents ({total_chunks} chunks total)")
    print("  You can now query the knowledge base at POST /api/v1/rag/query")


if __name__ == "__main__":
    asyncio.run(main())
