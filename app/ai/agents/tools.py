# WHY THIS FILE EXISTS:
# Defines the concrete tools the agent can invoke.
# Each tool is an async function with a clear signature.
# Tools are pure functions: given input, produce output. No side effects beyond logging.
#
# This design makes it trivial to add new tools:
# 1. Add an async function here
# 2. Add a ToolName enum value in models/agent.py
# 3. Add a routing case in orchestrator.py

from app.ai.llm.base import BaseLLM
from app.ai.prompts.templates import (
    DIRECT_ANSWER_SYSTEM_PROMPT,
    DIRECT_ANSWER_USER_PROMPT_TEMPLATE,
    SUMMARIZE_SYSTEM_PROMPT,
    SUMMARIZE_USER_PROMPT_TEMPLATE,
)
from app.ai.rag.pipeline import RAGPipeline
from app.core.logging import get_logger
from app.models.agent import ToolInfo, ToolName
from app.models.chat import ChatMessage

logger = get_logger(__name__)

# ── Tool Metadata ─────────────────────────────────────────────────────────────
TOOL_REGISTRY = [
    ToolInfo(
        name=ToolName.RETRIEVAL,
        description="Search the enterprise knowledge base for relevant information",
        when_to_use="When the user asks a factual question about documented topics (e.g., how-tos, policies, technical specs)",
    ),
    ToolInfo(
        name=ToolName.SUMMARIZE,
        description="Summarize a long piece of text into key points",
        when_to_use="When the user provides text and asks for a summary, key points, or overview",
    ),
    ToolInfo(
        name=ToolName.DIRECT_ANSWER,
        description="Answer conversational or general questions directly without document lookup",
        when_to_use="For greetings, capability questions, simple factual questions not requiring documents",
    ),
]


# ── Tool Implementations ──────────────────────────────────────────────────────

async def run_retrieval_tool(query: str, rag_pipeline: RAGPipeline) -> str:
    """
    Retrieval tool: runs the full RAG pipeline and returns the answer.

    Note: The RAG pipeline already applies guardrails internally, so we
    don't need to duplicate those checks here.
    """
    logger.info("agent_tool_retrieval", query_preview=query[:80])
    response = await rag_pipeline.run(query=query)
    return response.answer


async def run_summarize_tool(text: str, llm: BaseLLM) -> str:
    """
    Summarization tool: summarizes the provided text using the LLM.
    """
    logger.info("agent_tool_summarize", text_length=len(text))
    messages = [
        ChatMessage(role="system", content=SUMMARIZE_SYSTEM_PROMPT),
        ChatMessage(role="user", content=SUMMARIZE_USER_PROMPT_TEMPLATE.format(text=text)),
    ]
    response = await llm.complete(messages)
    return response.content


async def run_direct_answer_tool(query: str, llm: BaseLLM) -> str:
    """
    Direct answer tool: answers simple conversational queries without retrieval.
    """
    logger.info("agent_tool_direct", query_preview=query[:80])
    messages = [
        ChatMessage(role="system", content=DIRECT_ANSWER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=DIRECT_ANSWER_USER_PROMPT_TEMPLATE.format(query=query)),
    ]
    response = await llm.complete(messages)
    return response.content
