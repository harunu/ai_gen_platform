# WHY THIS FILE EXISTS:
# Text cleaning utilities used before embedding and before sending to LLM.
# Clean text improves embedding quality and reduces token waste.

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """
    Replace all runs of whitespace (spaces, tabs, newlines) with a single space
    and strip leading/trailing whitespace.
    """
    return re.sub(r"\s+", " ", text).strip()


def remove_control_characters(text: str) -> str:
    """
    Remove non-printable control characters (except common whitespace like \\n, \\t).
    These can appear in PDFs and cause encoding issues.
    """
    return "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", "\r")
    )


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """Truncate text to max_chars, appending a suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline:
    1. Remove control characters
    2. Normalize Unicode to NFC form (canonical composition)
    3. Normalize whitespace

    Apply this before chunking or embedding to ensure consistent input.
    """
    text = remove_control_characters(text)
    text = unicodedata.normalize("NFC", text)
    text = normalize_whitespace(text)
    return text


def count_words(text: str) -> int:
    """Return approximate word count (split on whitespace)."""
    return len(text.split())


def estimate_tokens(text: str) -> int:
    """
    Rough token count estimate: ~4 characters per token (OpenAI rule of thumb).
    This is an approximation; use a proper tokenizer for precise counts.
    """
    return max(1, len(text) // 4)
