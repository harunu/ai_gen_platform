# WHY THIS FILE EXISTS:
# Builds source citations from retrieved chunks.
# Citations are appended to RAG answers so users can verify claims
# against original documents — essential for enterprise trust.

from typing import List

from app.ai.prompts.templates import CITATION_ITEM_TEMPLATE, CITATION_TEMPLATE
from app.models.rag import SourceCitation


def build_citation_text(citations: List[SourceCitation]) -> str:
    """
    Build a formatted citation block to append to RAG answers.

    Args:
        citations: List of SourceCitation objects (already filtered and ranked).

    Returns:
        A formatted citation block string, e.g.:
        ---
        Sources:
        [1] Enterprise AI Guide (similarity: 0.87)
        [2] RAG Concepts (similarity: 0.72)
    """
    if not citations:
        return ""

    citation_lines = []
    for i, citation in enumerate(citations, start=1):
        title = citation.title or citation.source
        citation_lines.append(
            CITATION_ITEM_TEMPLATE.format(
                index=i,
                title=title,
                score=citation.similarity_score,
            )
        )

    return CITATION_TEMPLATE.format(citations="\n".join(citation_lines))


def build_context_block(citations: List[SourceCitation]) -> str:
    """
    Build the context block injected into the RAG prompt.

    Each chunk is labeled with its source for the LLM to reference.
    Format:
        [Document 1 - Enterprise AI Guide]
        <chunk text>

        [Document 2 - RAG Concepts]
        <chunk text>
    """
    if not citations:
        return "No relevant documents found."

    parts = []
    for i, citation in enumerate(citations, start=1):
        title = citation.title or citation.source
        parts.append(f"[Document {i} - {title}]\n{citation.content}")

    return "\n\n".join(parts)


def compute_confidence_hint(citations: List[SourceCitation]) -> str:
    """
    Compute a human-readable confidence hint based on the top similarity score.

    Thresholds are heuristic — tune based on your embedding model and data.
    - "high"   : top score >= 0.8 (very relevant chunk found)
    - "medium" : top score >= 0.5 (somewhat relevant)
    - "low"    : top score < 0.5 or no results (may be hallucinating)
    """
    if not citations:
        return "low"
    top_score = citations[0].similarity_score
    if top_score >= 0.8:
        return "high"
    elif top_score >= 0.5:
        return "medium"
    return "low"
