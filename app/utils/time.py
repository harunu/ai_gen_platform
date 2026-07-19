# WHY THIS FILE EXISTS:
# Centralizes timestamp helpers.
# Always use timezone-aware datetimes (UTC) to avoid subtle bugs
# when servers are in different timezones.

import time
from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return utcnow().isoformat()


def elapsed_ms(start: float) -> float:
    """
    Calculate elapsed milliseconds from a start time obtained via time.perf_counter().

    Usage:
        start = time.perf_counter()
        # ... do work ...
        ms = elapsed_ms(start)
    """
    return round((time.perf_counter() - start) * 1000, 2)
