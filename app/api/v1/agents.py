# WHY THIS FILE EXISTS:
# Agent endpoints: run the agent orchestrator and list available tools.

from fastapi import APIRouter, Depends, Request

from app.ai.agents.orchestrator import AgentOrchestrator
from app.ai.agents.tools import TOOL_REGISTRY
from app.ai.llm.base import BaseLLM
from app.ai.rag.pipeline import RAGPipeline
from app.ai.rag.retriever import RAGRetriever
from app.api.v1._providers import get_llm, get_vector_store_and_embeddings
from app.core.dependencies import CurrentUser
from app.core.rate_limit import AI_RATE_LIMIT, limiter
from app.models.agent import AgentRunRequest, AgentRunResponse, ToolListResponse

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/run", response_model=AgentRunResponse, summary="Run the agent")
@limiter.limit(AI_RATE_LIMIT)
async def run_agent(
    request: Request,
    body: AgentRunRequest,
    current_user: CurrentUser,
    deps=Depends(get_vector_store_and_embeddings),
    llm: BaseLLM = Depends(get_llm),
) -> AgentRunResponse:
    """
    Run the single-agent orchestrator for a user query.

    The agent:
    1. Decides which tool to use (retrieval, summarize, or direct answer)
    2. Executes the chosen tool
    3. Returns the answer with a full step trace for observability

    Requires JWT authentication.

    Example:
    ```json
    {"query": "What are the security considerations for enterprise AI?"}
    ```
    """
    vector_store, embeddings = deps
    retriever = RAGRetriever(vector_store=vector_store, embeddings=embeddings)
    rag_pipeline = RAGPipeline(retriever=retriever, llm=llm)
    orchestrator = AgentOrchestrator(llm=llm, rag_pipeline=rag_pipeline)

    return await orchestrator.run(request=body)


@router.get("/tools", response_model=ToolListResponse, summary="List available agent tools")
async def list_tools() -> ToolListResponse:
    """
    Return the list of tools available to the agent orchestrator.
    No authentication required — this is public metadata.
    """
    return ToolListResponse(tools=TOOL_REGISTRY)
