# WHY THIS FILE EXISTS:
# Tracks LLM token usage and estimated cost for every API call.
# Without cost tracking, LLM API bills become unpredictable in production.
# This module logs costs per call; wire it to a time-series DB for dashboards.

from app.core.logging import get_logger

logger = get_logger(__name__)

# Approximate cost per 1000 tokens in USD (update as pricing changes)
# Source: OpenAI pricing page (April 2025 approximations)
COST_PER_1K_TOKENS: dict = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Mock provider: no cost
    "mock-llm-v1": {"input": 0.0, "output": 0.0},
}

DEFAULT_COST = {"input": 0.001, "output": 0.002}  # Conservative estimate for unknown models


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate the cost of an LLM call in USD.

    Args:
        model: The model name (e.g. "gpt-4o-mini").
        prompt_tokens: Tokens in the input/prompt.
        completion_tokens: Tokens in the generated output.

    Returns:
        Estimated cost in USD (float).
    """
    costs = COST_PER_1K_TOKENS.get(model, DEFAULT_COST)
    input_cost = (prompt_tokens / 1000) * costs["input"]
    output_cost = (completion_tokens / 1000) * costs["output"]
    return round(input_cost + output_cost, 8)


def track_llm_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
) -> None:
    """
    Log token usage and estimated cost for an LLM call.

    This log line can be parsed by a log aggregator (Datadog, Loki, CloudWatch)
    to build cost dashboards and set budget alerts.

    In a production system, you'd also write to a time-series DB:
        metrics_client.increment("llm.tokens.prompt", prompt_tokens, tags={"model": model})
        metrics_client.gauge("llm.cost_usd", cost, tags={"model": model})
    """
    cost_usd = estimate_cost_usd(model, prompt_tokens, completion_tokens)
    total_tokens = prompt_tokens + completion_tokens

    logger.info(
        "llm_call_tracked",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=cost_usd,
        latency_ms=latency_ms,
    )
