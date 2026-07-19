# WHY THIS FILE EXISTS:
# Defines custom exception classes and global exception handlers.
# Centralizing exceptions means:
#   - Consistent error response shapes across all endpoints
#   - No stack traces leaked to clients (security best practice)
#   - All errors logged in one place with full context

from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from jose import JWTError

from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Custom Exception Hierarchy ────────────────────────────────────────────────

class AppBaseException(Exception):
    """
    Base class for all application-level exceptions.
    Carries an HTTP status code and a machine-readable error code.
    """
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class NotFoundError(AppBaseException):
    def __init__(self, resource: str, resource_id: str = ""):
        super().__init__(
            message=f"{resource} not found" + (f": {resource_id}" if resource_id else ""),
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
        )


class AuthenticationError(AppBaseException):
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_REQUIRED",
        )


class AuthorizationError(AppBaseException):
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="FORBIDDEN",
        )


class ValidationError(AppBaseException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class RateLimitError(AppBaseException):
    def __init__(self):
        super().__init__(
            message="Rate limit exceeded. Please slow down.",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
        )


class LLMError(AppBaseException):
    def __init__(self, message: str = "LLM service error"):
        super().__init__(
            message=message,
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code="LLM_ERROR",
        )


class VectorStoreError(AppBaseException):
    def __init__(self, message: str = "Vector store error"):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="VECTOR_STORE_ERROR",
        )


class GuardrailError(AppBaseException):
    def __init__(self, message: str, error_code: str = "GUARDRAIL_REJECTED"):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=error_code,
        )


def _error_response(
    status_code: int,
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """Build a consistent error response body."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": error_code,
                "message": message,
                "details": details or {},
            }
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers on the FastAPI application.
    Call this once during app startup in main.py.
    """

    @app.exception_handler(AppBaseException)
    async def handle_app_exception(request: Request, exc: AppBaseException) -> JSONResponse:
        logger.warning(
            "app_exception",
            error_code=exc.error_code,
            message=exc.message,
            path=request.url.path,
        )
        return _error_response(exc.status_code, exc.error_code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic validation errors: extract field-level messages
        errors = [
            {"field": " -> ".join(str(loc) for loc in e["loc"]), "message": e["msg"]}
            for e in exc.errors()
        ]
        logger.warning("request_validation_error", errors=errors, path=request.url.path)
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "VALIDATION_ERROR",
            "Request validation failed",
            {"errors": errors},
        )

    @app.exception_handler(JWTError)
    async def handle_jwt_error(request: Request, exc: JWTError) -> JSONResponse:
        logger.warning("jwt_error", error=str(exc), path=request.url.path)
        return _error_response(
            status.HTTP_401_UNAUTHORIZED,
            "INVALID_TOKEN",
            "Invalid or expired token",
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        # Log the FULL traceback internally, but return a safe generic message to client.
        # NEVER leak internal details (stack traces, DB queries, etc.) to API consumers.
        logger.error(
            "unhandled_exception",
            exc_info=exc,
            path=request.url.path,
            method=request.method,
        )
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "INTERNAL_ERROR",
            "An unexpected error occurred. Please try again later.",
        )
