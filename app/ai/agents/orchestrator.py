# WHY THIS FILE EXISTS:
# The agent orchestrator: decides which tool to use for a given query,
# executes it, and returns the result with a full step trace.
#
# Pattern: Single-agent decision loop
#   1. Receive user query
#   2. Call LLM to decide which tool to use (the "router" step)
#   3. Execute the chosen tool
#   4. Return answer + step trace
#
# This is simpler than a ReAct (Reason+Act) loop but covers most enterprise use cases.
# Extend to multi-step by wrapping steps 2-3 in a loop with a max_steps guard.

import json
import time
from typing import Optional

from app.ai.agents.memory import ConversationMemory
from app.ai.agents.tools import (
    TOOL_REGISTRY,
    run_direct_answer_tool,
    run_retrieval_tool,
    run_summarize_tool,
)
from app.ai.guardrails.injection import check_prompt_injection
from app.ai.guardrails.pii import check_and_mask_pii
from app.ai.llm.base import BaseLLM
from app.ai.prompts.templates import AGENT_SYSTEM_PROMPT, AGENT_USER_PROMPT_TEMPLATE
from app.ai.rag.pipeline import RAGPipeline
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.agent import AgentRunRequest, AgentRunResponse, AgentStep, ToolName
from app.models.chat import ChatMessage

logger = get_logger(__name__)


class AgentOrchestrator:
    """
    Single-agent orchestrator.

    The agent uses the LLM itself to decide which tool to use ("LLM-as-router").
    This is more flexible than rule-based routing but costs one extra LLM call per query.

    An alternative: train a small classifier for tool routing — faster and cheaper,
    but less flexible for novel query types.
    """

    def __init__(self, llm: BaseLLM, rag_pipeline: RAGPipeline):
        self.llm = llm
        self.rag_pipeline = rag_pipeline
        self.settings = get_settings()

    async def _decide_tool(self, query: str) -> ToolName:
        """
        Call the LLM to decide which tool to use.

        Returns a ToolName enum value.
        Falls back to DIRECT_ANSWER if the LLM response is unparseable.
        """
        messages = [
            ChatMessage(role="system", content=AGENT_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=AGENT_USER_PROMPT_TEMPLATE.format(query=query),
            ),
        ]

        response = await self.llm.complete(messages, temperature=0.0)  # temperature=0: deterministic routing

        try:
            # The LLM should return pure JSON: {"tool": "...", "reasoning": "..."}
            data = json.loads(response.content.strip())
            tool_str = data.get("tool", "DIRECT_ANSWER").upper()
            return ToolName(tool_str.lower())
        except (json.JSONDecodeError, ValueError, KeyError):
            # If the LLM returns malformed JSON, default to direct answer
            logger.warning("agent_tool_decision_parse_failed", raw_response=response.content[:200])
            return ToolName.DIRECT_ANSWER

    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        """
        Execute the agent for a user request.

        Args:
            request: AgentRunRequest with query and optional history.

        Returns:
            AgentRunResponse with answer and full step trace.
        """
        total_start = time.perf_counter()
        steps = []

        logger.info("agent_run_start", query_preview=request.query[:80])

        # ── Guardrails ───────────────────────────────────────────────────────
        check_prompt_injection(request.query, context="agent_query")
        clean_query = check_and_mask_pii(request.query)

        # ── Memory ───────────────────────────────────────────────────────────
        # Load history from the request (last N turns for context)
        memory = ConversationMemory()
        memory.load_from_history(request.history)

        # ── Tool Selection ───────────────────────────────────────────────────
        step_start = time.perf_counter()

        # If the caller explicitly forces a tool, skip the LLM routing call
        if request.force_tool:
            tool = request.force_tool
            reasoning = f"Tool forced by caller: {tool.value}"
        else:
            tool = await self._decide_tool(clean_query)
            reasoning = f"LLM selected {tool.value} for this query type"

        decision_latency = round((time.perf_counter() - step_start) * 1000, 2)

        # ── Tool Execution ───────────────────────────────────────────────────
        tool_start = time.perf_counter()

        if tool == ToolName.RETRIEVAL:
            tool_output = await run_retrieval_tool(clean_query, self.rag_pipeline)
        elif tool == ToolName.SUMMARIZE:
            tool_output = await run_summarize_tool(clean_query, self.llm)
        else:  # DIRECT_ANSWER
            tool_output = await run_direct_answer_tool(clean_query, self.llm)

        tool_latency = round((time.perf_counter() - tool_start) * 1000, 2)

        # Record the step for the trace
        steps.append(AgentStep(
            step_number=1,
            tool_used=tool,
            tool_input=clean_query,
            tool_output=tool_output[:500] + "..." if len(tool_output) > 500 else tool_output,
            reasoning=reasoning,
            latency_ms=tool_latency,
        ))

        # Update memory
        memory.add_user_message(clean_query)
        memory.add_assistant_message(tool_output)

        total_latency = round((time.perf_counter() - total_start) * 1000, 2)

        logger.info(
            "agent_run_complete",
            tool_used=tool.value,
            total_steps=len(steps),
            latency_ms=total_latency,
        )

        return AgentRunResponse(
            answer=tool_output,
            steps=steps,
            total_steps=len(steps),
            tool_used=tool,
            latency_ms=total_latency,
        )
