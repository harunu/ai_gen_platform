# WHY THIS FILE EXISTS:
# Business logic for the document ingestion endpoint.
# Validates input, delegates to DocumentRepository, and returns a structured response.

import time
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.rag import IngestRequest, IngestResponse
from app.repositories.document_repository import DocumentRepository

logger = get_logger(__name__)


class DocumentService:
    """Handles document ingestion requests from the API."""

    def __init__(self, document_repository: DocumentRepository):
        self.repo = document_repository

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        """
        Ingest a document from the API request.

        Accepts either:
        - `content`: raw text provided directly in the request body
        - `file_path`: a path relative to /data that will be read from disk

        Both are mutually exclusive; content takes priority.
        """
        start = time.perf_counter()

        if request.content:
            # Content provided directly in the request
            doc_id, num_chunks = await self.repo.ingest(
                content=request.content,
                metadata=request.metadata,
            )
        elif request.file_path:
            # Read from the mounted /data directory
            # Resolve to prevent path traversal attacks
            data_root = Path("/app/data").resolve()
            requested = (data_root / request.file_path).resolve()

            if not str(requested).startswith(str(data_root)):
                raise ValueError(f"Path traversal detected: {request.file_path}")
            if not requested.exists():
                raise FileNotFoundError(f"File not found: {request.file_path}")

            doc_id, num_chunks = await self.repo.ingest_from_file(
                file_path=str(requested),
                metadata=request.metadata,
            )
        else:
            raise ValueError("Either 'content' or 'file_path' must be provided")

        latency = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "document_service_ingest_complete",
            document_id=doc_id,
            num_chunks=num_chunks,
            latency_ms=latency,
        )

        return IngestResponse(
            document_id=doc_id,
            chunks_created=num_chunks,
            metadata=request.metadata,
            latency_ms=latency,
        )
