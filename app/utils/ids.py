# WHY THIS FILE EXISTS:
# Centralizes ID and correlation ID generation.
# Using one place for this logic means it's easy to swap UUID v4 for
# something like ULID (sortable) without touching every callsite.

import uuid


def generate_id() -> str:
    """Generate a random UUID v4 string (without hyphens for compactness)."""
    return uuid.uuid4().hex


def generate_request_id() -> str:
    """Generate a full UUID v4 string with hyphens (standard format for request IDs)."""
    return str(uuid.uuid4())


def generate_document_id(source: str) -> str:
    """
    Generate a deterministic document ID from a source string.
    Using UUID v5 (name-based): the same source always produces the same ID.
    This prevents duplicate ingestion of the same document.
    """
    # UUID v5 with DNS namespace + source string → deterministic UUID
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, source))
