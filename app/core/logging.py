# WHY THIS FILE EXISTS:
# Configures structlog for structured JSON logging throughout the application.
# Structured logging means every log line is a JSON object with consistent fields
# (timestamp, level, service, request_id, message, etc.) instead of free-form text.
# This makes logs trivially searchable in tools like Datadog, Loki, or CloudWatch.

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger


def add_service_name(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """
    Structlog processor: injects the 'service' field into every log entry.
    Processors are functions that transform the event dict before rendering.
    They run in the order listed in structlog.configure(processors=[...]).
    """
    event_dict["service"] = "ai_gen_platform"
    return event_dict


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog and the standard library logging to output JSON.

    structlog works alongside Python's standard logging module:
    - structlog handles the structured formatting
    - stdlib logging handles the final output (handlers, levels)
    Both are configured here so they work together seamlessly.
    """

    # --- Standard library logging setup ---
    # Set the root logger level so stdlib loggers (e.g. uvicorn, httpx)
    # also respect our configured level.
    logging.basicConfig(
        format="%(message)s",   # structlog will format the message itself
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    # Silence overly verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    # --- structlog configuration ---
    # Processors are applied left-to-right to transform the event dict.
    structlog.configure(
        processors=[
            # Add log level (e.g. "info", "error") to the event dict
            structlog.stdlib.add_log_level,
            # Add ISO 8601 timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Add our custom service name field
            add_service_name,
            # Render exception info as a structured dict (not a raw traceback string)
            structlog.processors.format_exc_info,
            # Render the final event dict as a JSON string
            structlog.processors.JSONRenderer(),
        ],
        # Use Python's standard logging under the hood for actual output
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,  # Performance: cache the bound logger
    )


def get_logger(name: str = __name__) -> Any:
    """
    Get a structlog logger bound with a module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("document_ingested", doc_id="abc", chunks=12)

    The resulting JSON log line looks like:
        {"event": "document_ingested", "doc_id": "abc", "chunks": 12,
         "level": "info", "timestamp": "2025-01-01T00:00:00Z", "service": "ai_gen_platform"}
    """
    return structlog.get_logger(name)
