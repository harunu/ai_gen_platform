# WHY THIS FILE EXISTS:
# Orchestrates the full RAG pipeline end-to-end.
# This is the most important file in the AI layer — read it to understand
# how retrieval, context assembly, guardrails, and LLM generation connect.
#
# Pipeline flow:
#   query → guardrails → embed → retrieve → assemble_context → LLM → validate → cite → return

import time
from typing import List, Optional

from app.ai.guardrails.injection import check_prompt_injection
from app.ai.guardrails.moderation import moderate_input, moderate_output
from app.ai.guardrails.output_validator import validate_llm_output
from app.ai.guardrails.pii import check_and_mask_pii
from app.ai.llm.base import BaseLLM
from app.ai.prompts.templates import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE
from app.ai.rag.citation import build_citation_text, build_context_block, compute_confidence_hint
from app.ai.rag.retriever import RAGRetriever
from app.ai.telemetry.cost_tracker import track_llm_call
from app.core.exceptions import GuardrailError, LLMError
from app.core.logging import get_logger
from app.models.chat import ChatMessage
from app.models.rag import RAGQueryResponse, SourceCitation

logger = get_logger(__name__)


class RAGPipeline:
    """
    Full RAG orchestration pipeline.

    Responsibilities:
    - Pre-process user query (guardrails, PII masking)
    - Retrieve relevant document chunks
    - Assemble LLM prompt with context
    - Call LLM and validate response
    - Build response with citations

    Dependencies are injected (not hardcoded) so this class is testable
    with mock implementations.
    """

    def __init__(self, retriever: RAGRetriever, llm: BaseLLM):
        self.retriever = retriever
        self.llm = llm

    async def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_tags: Optional[List[str]] = None,
    ) -> RAGQueryResponse:
        """
        Execute the full RAG pipeline for a user query.

        Args:
            query: The user's question.
            top_k: Number of chunks to retrieve.
            similarity_threshold: Minimum similarity to include a chunk.
            filter_tags: Optional metadata tags to filter retrieved chunks.

        Returns:
            RAGQueryResponse with answer, sources, confidence, and latency.
        """
        total_start = time.perf_counter()

        logger.info("rag_pipeline_start", query_preview=query[:80])

        # ── Step 1: Input Guardrails ─────────────────────────────────────────
        # Order matters: check injection BEFORE masking PII, so injected
        # instructions aren't accidentally allowed through by the masker.
        check_prompt_injection(query, context="rag_query")

        # Mask PII in the query before it's embedded or sent to LLM
        # This prevents personal data from being stored in the vector index
        # or sent to external LLM APIs.
        clean_query = check_and_mask_pii(query)

        # Content moderation check (stub — wire in your moderation API)
        is_safe = await moderate_input(clean_query)
        if not is_safe:
            raise GuardrailError("Query flagged by content moderation", "CONTENT_MODERATION_FAILED")

        # ── Step 2: Retrieval ────────────────────────────────────────────────
        retrieval_start = time.perf_counter()
        citations: List[SourceCitation] = await self.retriever.retrieve(
            query=clean_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter_tags=filter_tags,
        )
        retrieval_latency = round((time.perf_counter() - retrieval_start) * 1000, 2)

        # ── Step 3: Context Assembly ─────────────────────────────────────────
        # Build the context block from retrieved chunks.
        # The context is injected into the user message (not the system prompt)
        # so the LLM sees it as "provided information" rather than "instructions".
        context_block = build_context_block(citations)

        user_message_content = RAG_USER_PROMPT_TEMPLATE.format(
            context=context_block,
            question=clean_query,
        )

        messages = [
            ChatMessage(role="system", content=RAG_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_message_content),
        ]

        # ── Step 4: LLM Generation ───────────────────────────────────────────
        llm_start = time.perf_counter()
        llm_response = await self.llm.complete(messages)
        llm_latency = round((time.perf_counter() - llm_start) * 1000, 2)

        # ── Step 5: Cost Tracking ────────────────────────────────────────────
        track_llm_call(
            model=llm_response.model,
            prompt_tokens=llm_response.usage.prompt_tokens,
            completion_tokens=llm_response.usage.completion_tokens,
            latency_ms=llm_latency,
        )

        # ── Step 6: Output Validation ────────────────────────────────────────
        validated_answer = validate_llm_output(llm_response.content, context="rag_answer")

        # ── Step 7: Output Moderation ─────────────────────────────────────────
        output_safe = await moderate_output(validated_answer)
        if not output_safe:
            raise GuardrailError("Response flagged by content moderation", "OUTPUT_MODERATION_FAILED")

        # ── Step 8: Citation Building ─────────────────────────────────────────
        citation_text = build_citation_text(citations)
        final_answer = validated_answer + citation_text if citation_text else validated_answer
        confidence = compute_confidence_hint(citations)

        total_latency = round((time.perf_counter() - total_start) * 1000, 2)

        logger.info(
            "rag_pipeline_complete",
            query_preview=query[:80],
            num_sources=len(citations),
            confidence=confidence,
            total_latency_ms=total_latency,
            retrieval_latency_ms=retrieval_latency,
            llm_latency_ms=llm_latency,
        )

        return RAGQueryResponse(
            answer=final_answer,
            sources=citations,
            confidence_hint=confidence,
            query=query,
            latency_ms=total_latency,
            retrieval_latency_ms=retrieval_latency,
            llm_latency_ms=llm_latency,
        )
