# WHY THIS FILE EXISTS:
# Retrieval quality evaluation skeleton.
# Measuring RAG quality is critical for production systems — without metrics,
# you don't know if your chunking, embedding, or retrieval settings are good.
#
# Key RAG metrics:
# - Retrieval Precision: Of the chunks retrieved, how many were actually relevant?
# - Retrieval Recall: Of all relevant chunks, how many did we retrieve?
# - Answer Faithfulness: Does the LLM answer stay within the retrieved context?
# - Answer Relevance: Does the answer actually address the question?
#
# This skeleton shows the structure. Wire in a proper eval framework like
# RAGAS (https://github.com/explodinggradients/ragas) for production use.

from dataclasses import dataclass
from typing import List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalEvalResult:
    """Results of a single retrieval evaluation."""
    query: str
    retrieved_chunk_ids: List[str]
    relevant_chunk_ids: List[str]      # Ground truth: what should have been retrieved
    precision: float                    # TP / (TP + FP)
    recall: float                       # TP / (TP + FN)
    f1: float                           # Harmonic mean of precision and recall


@dataclass
class AnswerEvalResult:
    """Results of an answer quality evaluation."""
    query: str
    answer: str
    context_used: str
    faithfulness_score: float           # 0–1: does the answer stay within the context?
    relevance_score: float              # 0–1: does the answer address the question?


def compute_retrieval_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> RetrievalEvalResult:
    """
    Compute precision, recall, and F1 for a single retrieval result.

    Args:
        retrieved_ids: Chunk IDs returned by the retriever.
        relevant_ids: Ground-truth relevant chunk IDs (from a labeled dataset).
    """
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)

    true_positives = len(retrieved_set & relevant_set)

    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return RetrievalEvalResult(
        query="",
        retrieved_chunk_ids=retrieved_ids,
        relevant_chunk_ids=relevant_ids,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )


async def evaluate_rag_batch(
    eval_dataset: List[dict],  # [{"query": str, "relevant_ids": List[str]}]
    retriever,
) -> dict:
    """
    Run retrieval evaluation over a batch of labeled queries.

    Args:
        eval_dataset: List of {"query": ..., "relevant_ids": [...]} dicts.
        retriever: RAGRetriever instance to evaluate.

    Returns:
        Aggregate metrics: mean precision, recall, F1.

    TODO: Replace with RAGAS or a custom eval framework for production.
    """
    results = []
    for item in eval_dataset:
        citations = await retriever.retrieve(query=item["query"])
        retrieved_ids = [c.chunk_id for c in citations]
        result = compute_retrieval_metrics(retrieved_ids, item["relevant_ids"])
        result.query = item["query"]
        results.append(result)

    if not results:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    avg_precision = sum(r.precision for r in results) / len(results)
    avg_recall = sum(r.recall for r in results) / len(results)
    avg_f1 = sum(r.f1 for r in results) / len(results)

    metrics = {
        "precision": round(avg_precision, 4),
        "recall": round(avg_recall, 4),
        "f1": round(avg_f1, 4),
        "num_queries": len(results),
    }

    logger.info("rag_eval_complete", **metrics)
    return metrics
