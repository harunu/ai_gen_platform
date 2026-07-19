# WHY THIS FILE EXISTS:
# Middleware sits between the HTTP server and your route handlers.
# Every incoming request passes through middleware BEFORE reaching any endpoint.
# Every outgoing response passes through middleware AFTER the endpoint runs.
# We use middleware for cross-cutting concerns: request IDs, timing, CORS, security headers.

import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Assigns a unique X-Request-ID to every request.

    Why this matters:
    - When a user reports a bug with their request ID, you can grep logs for it
    - Correlates log lines across multiple services in distributed systems
    - The ID is injected into all log lines via request.state so downstream
      code can include it without passing it explicitly
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Accept an existing request ID (e.g. from a load balancer) or generate a new one
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store on request.state so route handlers and services can access it
        request.state.request_id = request_id

        # Process the request and measure latency
        start_time = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Add request ID and timing to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(latency_ms)

        # Log every request: method, path, status, latency
        logger.info(
            "http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to every response.

    These headers instruct browsers to apply additional protections:
    - X-Content-Type-Options: Prevents browsers from MIME-sniffing responses
      (stops e.g. a text/plain response being executed as JavaScript)
    - X-Frame-Options: Prevents clickjacking by disallowing iframes
    - X-XSS-Protection: Legacy XSS filter (modern browsers use CSP instead)
    - Referrer-Policy: Controls what URL is sent in the Referer header
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Remove the Server header to avoid revealing the web framework version
        if "server" in response.headers:
            del response.headers["server"]
        return response


def register_middleware(app: FastAPI) -> None:
    """
    Register all middleware on the FastAPI application.

    ORDER MATTERS: Middleware is applied in reverse registration order.
    The LAST middleware registered wraps the OUTERMOST layer.
    Register in order: innermost → outermost.
    """
    settings = get_settings()

    # 1. Request ID + timing (innermost — runs close to the route handler)
    app.add_middleware(RequestIDMiddleware)

    # 2. Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # 3. CORS (outermost — browsers check CORS before sending the actual request)
    # CORSMiddleware handles OPTIONS preflight requests automatically.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],     # Allow all HTTP methods
        allow_headers=["*"],     # Allow all request headers
    )
