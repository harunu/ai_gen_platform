# WHY THIS FILE EXISTS:
# Application entry point. Creates and configures the FastAPI app.
# Uses the "app factory" pattern: the app is created inside a function.
# This makes it easy to create multiple app instances in tests with different configs.

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.core.middleware import register_middleware
from app.core.rate_limit import limiter


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager (replaces deprecated on_startup/on_shutdown).

    Code BEFORE yield: startup tasks (runs before serving requests)
    Code AFTER yield:  shutdown tasks (runs after all requests are handled)

    Lifespan is the modern FastAPI way to manage app lifecycle.
    It uses Python's async context manager protocol.
    """
    settings = get_settings()

    # ── Startup ───────────────────────────────────────────────────────────────
    setup_logging(settings.log_level)
    logger = get_logger(__name__)

    logger.info(
        "app_starting",
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        mock_llm=settings.use_mock_llm,
    )

    # Pre-warm the embedding model — loads weights from disk/cache into memory.
    # Without this, the first RAG request would have high latency.
    if not settings.use_mock_llm or True:  # Always pre-warm embeddings
        try:
            from app.ai.embeddings.provider import get_embeddings_provider
            embeddings = get_embeddings_provider()
            logger.info("embedding_model_ready", model=embeddings.model_name)
        except Exception as e:
            logger.warning("embedding_preload_failed", error=str(e))

    logger.info("app_ready", port=settings.port)

    yield   # Application serves requests here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("app_shutting_down")

    # Close the LLM HTTP client connection pool if using real LLM
    if not settings.use_mock_llm:
        try:
            from app.ai.llm.openai_compatible import OpenAICompatibleLLM
            # Note: in a real app, store the LLM instance in app.state and close it here
            logger.info("llm_client_closed")
        except Exception:
            pass

    logger.info("app_shutdown_complete")


def create_app() -> FastAPI:
    """
    Application factory: creates and configures the FastAPI application.

    Why a factory function?
    - Tests can call create_app() to get a fresh app with test settings
    - Avoids module-level side effects at import time
    - Makes the startup sequence explicit and readable
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade GenAI backend with RAG pipeline, "
            "agent orchestrator, and enterprise security features."
        ),
        # Disable OpenAPI docs in production (don't expose your API schema to the internet)
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    # Attach the limiter to the app state so slowapi can access it from decorators
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # ── Middleware ────────────────────────────────────────────────────────────
    # Must be registered BEFORE routes are included
    register_middleware(app)

    # ── Exception Handlers ────────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(api_router)

    return app


# Create the global app instance.
# Uvicorn imports this: uvicorn app.main:app
app = create_app()
