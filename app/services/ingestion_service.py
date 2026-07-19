# WHY THIS FILE EXISTS:
# Bulk ingestion service: reads all documents from a directory and ingests them.
# Used by the `make ingest` command (scripts/ingest_documents.py).
# Separated from DocumentService to keep the API service focused on single-document operations.

import asyncio
import time
from pathlib import Path
from typing import List, Tuple

from app.core.logging import get_logger
from app.models.rag import DocumentMetadata
from app.repositories.document_repository import DocumentRepository

logger = get_logger(__name__)

# Supported file extensions for document ingestion
SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst"}


class IngestionService:
    """Bulk document ingestion from a directory."""

    def __init__(self, document_repository: DocumentRepository):
        self.repo = document_repository

    async def ingest_directory(self, directory: str) -> List[Tuple[str, int]]:
        """
        Ingest all supported documents from a directory.

        Args:
            directory: Path to the directory containing documents.

        Returns:
            List of (document_id, num_chunks) tuples for each ingested document.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Discover all supported files
        files = [
            f for f in dir_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not files:
            logger.warning("no_documents_found", directory=directory)
            return []

        logger.info("ingestion_started", directory=directory, file_count=len(files))
        start = time.perf_counter()

        results = []
        errors = []

        for file_path in files:
            try:
                metadata = DocumentMetadata(
                    source=str(file_path.relative_to(dir_path)),
                    title=file_path.stem.replace("_", " ").title(),
                )
                doc_id, num_chunks = await self.repo.ingest_from_file(
                    file_path=str(file_path),
                    metadata=metadata,
                )
                results.append((doc_id, num_chunks))
                logger.info(
                    "file_ingested",
                    file=file_path.name,
                    document_id=doc_id,
                    chunks=num_chunks,
                )
            except Exception as e:
                logger.error("file_ingest_failed", file=str(file_path), error=str(e))
                errors.append(str(file_path))

        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "ingestion_complete",
            total_files=len(files),
            successful=len(results),
            failed=len(errors),
            latency_ms=elapsed,
        )

        return results
