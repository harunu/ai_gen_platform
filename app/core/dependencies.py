# WHY THIS FILE EXISTS:
# FastAPI's dependency injection system lets you declare shared logic
# (auth checks, DB connections, service instances) as "dependencies".
# Dependencies are reusable, testable, and automatically called by FastAPI
# before executing the route handler.
#
# This file is the central "hub" for all injectable dependencies.
# Route handlers import from here, not from scattered modules.

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.core.security import decode_access_token
from app.core.logging import get_logger

logger = get_logger(__name__)

# HTTPBearer parses the "Authorization: Bearer <token>" header.
# auto_error=False means it won't raise automatically; we handle errors manually
# to return our consistent error shape.
bearer_scheme = HTTPBearer(auto_error=False)


# ── Settings Dependency ───────────────────────────────────────────────────────

def settings_dep() -> Settings:
    """Inject application settings."""
    return get_settings()

SettingsDep = Annotated[Settings, Depends(settings_dep)]


# ── JWT Authentication ────────────────────────────────────────────────────────

async def require_jwt(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_scheme)],
) -> str:
    """
    Dependency that validates a JWT Bearer token.

    Usage in a route:
        @router.post("/chat")
        async def chat(current_user: str = Depends(require_jwt)):
            ...  # current_user is the token subject (e.g. username)

    Raises AuthenticationError if:
    - No Authorization header is present
    - Token is malformed, expired, or has an invalid signature
    """
    if credentials is None:
        raise AuthenticationError("Authorization header missing")

    try:
        payload = decode_access_token(credentials.credentials)
        subject: str | None = payload.get("sub")
        if not subject:
            raise AuthenticationError("Token missing subject claim")
        return subject
    except JWTError as e:
        logger.warning("jwt_validation_failed", error=str(e))
        raise AuthenticationError("Invalid or expired token")


CurrentUser = Annotated[str, Depends(require_jwt)]


# ── API Key Authentication ────────────────────────────────────────────────────

async def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    settings: Settings = Depends(settings_dep),
) -> str:
    """
    Dependency that validates the X-API-Key header.

    API Keys are used for server-to-server calls (e.g. ingestion pipeline,
    internal services) where JWT is overkill.

    Raises AuthorizationError if the key is missing or invalid.
    """
    if x_api_key is None:
        raise AuthorizationError("X-API-Key header required")

    # Constant-time comparison prevents timing attacks:
    # An attacker cannot determine how many characters match by measuring response time.
    import hmac
    if not hmac.compare_digest(x_api_key, settings.api_key):
        logger.warning("invalid_api_key_attempt")
        raise AuthorizationError("Invalid API key")

    return x_api_key


APIKeyDep = Annotated[str, Depends(require_api_key)]
