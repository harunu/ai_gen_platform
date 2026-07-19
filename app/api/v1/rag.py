# WHY THIS FILE EXISTS:
# RAG endpoints: query the knowledge base and ingest new documents.
# Both endpoints require an API key (X-API-Key header) for server-to-server access.

from fastapi import APIRouter, Depends, Request

from app.ai.llm.base import BaseLLM
from app.ai.rag.pipeline import RAGPipeline
from app.ai.rag.retriever import RAGRetriever
from app.api.v1._providers import get_llm, get_vector_store_and_embeddings
from app.core.dependencies import APIKeyDep
from app.core.rate_limit import AI_RATE_LIMIT, limiter
from app.models.rag import IngestRequest, IngestResponse, RAGQueryRequest, RAGQueryResponse
from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_store import VectorStore
from app.services.document_service import DocumentService

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGQueryResponse, summary="Query the knowledge base")
@limiter.limit(AI_RATE_LIMIT)
async def rag_query(
    request: Request,
    body: RAGQueryRequest,
    _api_key: APIKeyDep,                    # Validates X-API-Key header
    deps=Depends(get_vector_store_and_embeddings),
    llm: BaseLLM = Depends(get_llm),
) -> RAGQueryResponse:
    """
    Run a RAG query against the document knowledge base.

    Flow: embed query → similarity search → assemble context → LLM → return answer + sources.

    Requires X-API-Key header.

    Example:
    ```json
    {"query": "What is retrieval-augmented generation?"}
    ```
    """
    vector_store, embeddings = deps
    retriever = RAGRetriever(vector_store=vector_store, embeddings=embeddings)
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    return await pipeline.run(
        query=body.query,
        top_k=body.top_k,
        similarity_threshold=body.similarity_threshold,
        filter_tags=body.filter_tags,
    )


@router.post("/ingest", response_model=IngestResponse, summary="Ingest a document")
@limiter.limit("5/minute")
async def ingest_document(
    request: Request,
    body: IngestRequest,
    _api_key: APIKeyDep,
    deps=Depends(get_vector_store_and_embeddings),
) -> IngestResponse:
    """
    Ingest a document into the vector knowledge base.

    Accepts either raw `content` (text) or a `file_path` relative to /data.
    The document is chunked, embedded, and stored in ChromaDB.

    Requires X-API-Key header.
    """
    vector_store, embeddings = deps
    repo = DocumentRepository(
        vector_store=vector_store,
        embeddings=embeddings,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
    )
    service = DocumentService(document_repository=repo)
    return await service.ingest(request=body)
