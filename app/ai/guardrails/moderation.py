# WHY THIS FILE EXISTS:
# Hook for content moderation before sending user input to the LLM.
# This stub integrates with an external moderation API (e.g. OpenAI Moderation,
# Azure Content Safety, or a custom model).
# Currently a passthrough — wire in your chosen moderation service here.

from app.core.logging import get_logger

logger = get_logger(__name__)


async def moderate_input(text: str) -> bool:
    """
    Check if user input is safe to process.

    Returns:
        True if the content is safe, False if it should be blocked.

    TODO: Replace the passthrough below with a real moderation API call:

    Example using OpenAI Moderation API:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/moderations",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": text},
            )
            result = response.json()
            return not result["results"][0]["flagged"]

    Example using Azure Content Safety:
        from azure.ai.contentsafety import ContentSafetyClient
        # ... Azure SDK call ...
    """
    # Current implementation: log the call and allow all content
    logger.debug("moderation_check", text_length=len(text), result="allowed_passthrough")
    return True


async def moderate_output(text: str) -> bool:
    """
    Check if LLM output is safe to return to the user.

    Returns:
        True if the output is safe, False if it should be blocked.

    TODO: Same as above — wire in your moderation service.
    """
    logger.debug("output_moderation_check", text_length=len(text), result="allowed_passthrough")
    return True
