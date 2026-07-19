# WHY THIS FILE EXISTS:
# Validates LLM output before returning it to the caller.
# Catches empty responses, truncated outputs, and oversized responses
# that would indicate model errors or abuse.

from app.core.exceptions import GuardrailError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Configuration constants
MIN_RESPONSE_LENGTH = 5         # Characters — fewer than this is likely an error
MAX_RESPONSE_LENGTH = 50_000    # Characters — more than this is likely runaway generation


def validate_llm_output(text: str, context: str = "llm_response") -> str:
    """
    Validate and return the LLM output, or raise GuardrailError.

    Checks:
    1. Not empty or whitespace-only
    2. Not shorter than minimum meaningful response
    3. Not longer than maximum allowed response

    Args:
        text: The raw LLM output text.
        context: Label for logging (e.g. "rag_answer", "agent_response").

    Returns:
        The stripped, validated text.

    Raises:
        GuardrailError: If validation fails.
    """
    stripped = text.strip() if text else ""

    if not stripped:
        logger.warning("output_validation_failed", reason="empty_response", context=context)
        raise GuardrailError(
            "The AI model returned an empty response. Please try again.",
            error_code="EMPTY_LLM_RESPONSE",
        )

    if len(stripped) < MIN_RESPONSE_LENGTH:
        logger.warning(
            "output_validation_failed",
            reason="response_too_short",
            length=len(stripped),
            context=context,
        )
        raise GuardrailError(
            "The AI model returned an unusually short response. Please try again.",
            error_code="RESPONSE_TOO_SHORT",
        )

    if len(stripped) > MAX_RESPONSE_LENGTH:
        logger.warning(
            "output_validation_failed",
            reason="response_too_long",
            length=len(stripped),
            context=context,
        )
        # Truncate rather than reject — still return value to the user
        stripped = stripped[:MAX_RESPONSE_LENGTH] + "\n\n[Response truncated]"

    logger.debug("output_validation_passed", length=len(stripped), context=context)
    return stripped
