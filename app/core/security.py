# WHY THIS FILE EXISTS:
# Handles JWT creation and validation.
# JWT (JSON Web Token) is a compact, signed token used to authenticate API requests.
# A token contains: header (algorithm) + payload (user claims) + signature.
# The signature is verified using the secret key — without it, tokens cannot be forged.

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
from jose import JWTError, jwt

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def create_access_token(
    subject: str,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a signed JWT access token.

    Args:
        subject: Usually the user ID or username (goes into the 'sub' claim).
        extra_claims: Optional additional key-value pairs to embed in the token.

    Returns:
        A signed JWT string (three base64url segments separated by dots).

    JWT claims used here:
        sub  — Subject (who this token is for)
        exp  — Expiration time (Unix timestamp)
        iat  — Issued at time
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.jwt_expire_minutes)

    # Build the payload (claims)
    payload: Dict[str, Any] = {
        "sub": subject,
        "exp": expire,
        "iat": now,
    }
    if extra_claims:
        payload.update(extra_claims)

    # Sign the token with the secret key using the configured algorithm
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    logger.info("access_token_created", subject=subject, expires_at=expire.isoformat())
    return token


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.

    Raises:
        JWTError: If the token is invalid, expired, or tampered with.

    Returns:
        The decoded payload dictionary.
    """
    settings = get_settings()
    # jwt.decode() verifies the signature AND the expiry automatically.
    # If the token has been tampered with or expired, it raises JWTError.
    payload = jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )
    return payload


def get_token_subject(token: str) -> Optional[str]:
    """
    Extract the subject (user ID) from a token without raising on failure.
    Returns None if the token is invalid or missing the 'sub' claim.
    """
    try:
        payload = decode_access_token(token)
        return payload.get("sub")
    except JWTError:
        return None


def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a plain-text password against a bcrypt hash.
    Returns True if they match.
    """
    return bcrypt.checkpw(plain.encode(), hashed.encode())
