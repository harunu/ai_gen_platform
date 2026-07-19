# WHY THIS FILE EXISTS:
# PII (Personally Identifiable Information) masking before sending text to an LLM.
# Regulations like GDPR and HIPAA require that personal data is not sent to
# external services without consent. This regex-based masker replaces PII
# with placeholder tokens so the LLM never sees real personal data.
# For production, replace or augment with a dedicated NER model (e.g. spaCy, Presidio).

import re
from typing import List, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)

# Each tuple: (pattern_name, compiled_regex, replacement_token)
# Patterns are applied in order; earlier patterns take priority.
PII_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[SSN_REDACTED]",
    ),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL_REDACTED]",
    ),
    (
        "phone_us",
        # Matches: (555) 867-5309, 555-867-5309, +1 555 867 5309, etc.
        re.compile(r"(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})"),
        "[PHONE_REDACTED]",
    ),
    (
        "credit_card",
        # Matches 13-19 digit card numbers (with optional dashes/spaces)
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        "[CARD_REDACTED]",
    ),
    (
        "ip_address",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "[IP_REDACTED]",
    ),
]


def mask_pii(text: str) -> Tuple[str, int]:
    """
    Apply all PII masking patterns to the text.

    Returns:
        (masked_text, count_of_replacements)
        The count lets callers decide whether to log a PII-found event.
    """
    total_replacements = 0
    for pattern_name, pattern, replacement in PII_PATTERNS:
        masked, count = pattern.subn(replacement, text)
        if count > 0:
            logger.info(
                "pii_masked",
                pattern=pattern_name,
                count=count,
            )
            total_replacements += count
            text = masked

    return text, total_replacements


def check_and_mask_pii(text: str) -> str:
    """
    Mask PII and return the sanitized text.
    Logs a summary if any PII was found.
    """
    masked, count = mask_pii(text)
    if count > 0:
        logger.warning("pii_found_and_masked", total_redactions=count)
    return masked
