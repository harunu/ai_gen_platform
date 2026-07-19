# WHY THIS FILE EXISTS:
# Prompt regression testing skeleton.
# When you modify a system prompt, you want to verify that existing queries
# still return equivalent (or better) answers. This is the prompt engineering
# equivalent of unit regression tests.
#
# Structure:
# 1. Store a "golden dataset" of (query, expected_answer_keywords) pairs
# 2. Run the current pipeline against each query
# 3. Check that expected keywords appear in the output
# 4. Flag regressions (queries that previously passed but now fail)

from dataclasses import dataclass
from typing import List

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTestCase:
    """A single test case for prompt regression testing."""
    test_id: str
    query: str
    expected_keywords: List[str]    # All of these must appear in the answer
    forbidden_keywords: List[str]   # None of these should appear in the answer
    description: str = ""


@dataclass
class PromptTestResult:
    """Result of running a single test case."""
    test_id: str
    query: str
    answer: str
    passed: bool
    missing_keywords: List[str]
    found_forbidden: List[str]
    failure_reason: str = ""


def evaluate_answer(
    answer: str,
    test_case: PromptTestCase,
) -> PromptTestResult:
    """
    Check if an answer passes the test case requirements.

    Args:
        answer: The LLM-generated answer to evaluate.
        test_case: The test case with expected/forbidden keyword lists.

    Returns:
        PromptTestResult indicating pass/fail with details.
    """
    answer_lower = answer.lower()

    missing = [kw for kw in test_case.expected_keywords if kw.lower() not in answer_lower]
    forbidden_found = [kw for kw in test_case.forbidden_keywords if kw.lower() in answer_lower]

    passed = (not missing) and (not forbidden_found)
    failure_reason = ""
    if missing:
        failure_reason += f"Missing keywords: {missing}. "
    if forbidden_found:
        failure_reason += f"Found forbidden keywords: {forbidden_found}."

    return PromptTestResult(
        test_id=test_case.test_id,
        query=test_case.query,
        answer=answer,
        passed=passed,
        missing_keywords=missing,
        found_forbidden=forbidden_found,
        failure_reason=failure_reason.strip(),
    )


async def run_prompt_regression_suite(
    test_cases: List[PromptTestCase],
    pipeline,  # RAGPipeline or any callable returning an answer
) -> dict:
    """
    Run a full regression suite against the current pipeline.

    Args:
        test_cases: List of PromptTestCase objects.
        pipeline: The pipeline to test (must have an async `run(query)` method).

    Returns:
        Summary dict with pass/fail counts and individual results.
    """
    results = []
    for case in test_cases:
        try:
            response = await pipeline.run(query=case.query)
            result = evaluate_answer(response.answer, case)
        except Exception as e:
            result = PromptTestResult(
                test_id=case.test_id,
                query=case.query,
                answer="",
                passed=False,
                missing_keywords=case.expected_keywords,
                found_forbidden=[],
                failure_reason=f"Pipeline error: {e}",
            )
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    summary = {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "failures": [
            {"test_id": r.test_id, "reason": r.failure_reason}
            for r in results if not r.passed
        ],
    }

    logger.info("prompt_regression_complete", **{k: v for k, v in summary.items() if k != "failures"})
    return summary
