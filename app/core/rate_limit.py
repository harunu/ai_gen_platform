# WHY THIS FILE EXISTS:
# Configures slowapi rate limiting for the application.
# Rate limiting prevents abuse: a single client cannot send unlimited requests
# and overwhelm the server or rack up huge LLM API bills.
# slowapi is a port of Flask-Limiter for FastAPI/Starlette.

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import get_settings

settings = get_settings()

# Limiter instance used throughout the app.
# key_func=get_remote_address: rate limits are per client IP address.
# In production behind a load balancer, replace with a function that reads
# the X-Forwarded-For header to get the real client IP.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],   # No default limit; set per-route with @limiter.limit(...)
)


def get_limiter() -> Limiter:
    """Dependency: returns the configured limiter instance."""
    return limiter


# Rate limit strings (read from config for easy env-based tuning):
# Format: "<count>/<period>" where period is second|minute|hour|day
AI_RATE_LIMIT = settings.rate_limit_ai        # e.g. "10/minute"
HEALTH_RATE_LIMIT = settings.rate_limit_health  # e.g. "60/minute"
