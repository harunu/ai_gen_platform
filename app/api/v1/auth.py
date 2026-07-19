# WHY THIS FILE EXISTS:
# Authentication endpoint: issues JWT tokens.
# In production, validate against a real user database.
# Here we use a hardcoded demo user for local development.
# Extend this by adding a users table (SQLite, Postgres) and hashed passwords.

from fastapi import APIRouter, Request

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError
from app.core.rate_limit import AI_RATE_LIMIT, limiter
from app.core.security import create_access_token, verify_password
from app.models.auth import TokenRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])

# Demo user store — replace with a real database in production.
# Passwords stored as bcrypt hashes (generated via: passlib.hash.bcrypt.hash("password"))
DEMO_USERS = {
    "alice": {
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "role": "admin",
    },
    "bob": {
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "role": "user",
    },
}


@router.post("/token", response_model=TokenResponse, summary="Issue JWT access token")
@limiter.limit("20/minute")
async def issue_token(request: Request, body: TokenRequest) -> TokenResponse:
    """
    Authenticate with username/password and receive a JWT access token.

    The token should be included in subsequent requests as:
        Authorization: Bearer <token>

    Demo credentials:
        username: alice, password: secret
        username: bob, password: secret
    """
    settings = get_settings()

    user = DEMO_USERS.get(body.username)
    if not user:
        # Use the same error message for missing users AND wrong passwords.
        # Leaking "user not found" vs "wrong password" helps attackers enumerate valid users.
        raise AuthenticationError("Invalid username or password")

    if not verify_password(body.password, user["hashed_password"]):
        raise AuthenticationError("Invalid username or password")

    token = create_access_token(
        subject=body.username,
        extra_claims={"role": user["role"]},
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.jwt_expire_minutes * 60,
    )
