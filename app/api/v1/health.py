# WHY THIS FILE EXISTS:
# Health check endpoints for liveness and readiness probes.
# Liveness (/health): Is the app process running? (Docker/K8s restarts on failure)
# Readiness (/health/ready): Is the app ready to serve traffic?
#   - Checks ChromaDB connectivity
#   - Checks LLM connectivity
# These are standard in cloud deployments and load balancer health checks.

import time

from fastapi import APIRouter, Depends, Request

from app.core.config import Settings, get_settings
from app.core.rate_limit import HEALTH_RATE_LIMIT, limiter
from app.models.common import HealthStatus

router = APIRouter(prefix="/health", tags=["health"])

# Track when the app started so we can report uptime
_START_TIME = time.time()


@router.get("", response_model=HealthStatus, summary="Liveness check")
@limiter.limit(HEALTH_RATE_LIMIT)
async def liveness(request: Request, settings: Settings = Depends(get_settings)) -> HealthStatus:
    """
    Liveness probe: returns 200 if the application process is running.

    Docker and Kubernetes restart the container if this returns a non-200 status.
    This endpoint should be FAST — no external calls. Just return OK.
    """
    return HealthStatus(
        status="ok",
        version=settings.app_version,
        uptime_seconds=round(time.time() - _START_TIME, 1),
        vector_db_status="not_checked",
        llm_status="not_checked",
        environment=settings.app_env,
    )


@router.get("/ready", response_model=HealthStatus, summary="Readiness check")
@limiter.limit(HEALTH_RATE_LIMIT)
async def readiness(request: Request, settings: Settings = Depends(get_settings)) -> HealthStatus:
    """
    Readiness probe: checks that all downstream dependencies are reachable.

    Load balancers stop sending traffic if this returns non-200.
    Check ChromaDB and LLM connectivity here.
    """
    from app.repositories.chroma_store import get_chroma_store
    from app.ai.llm.mock_provider import MockLLMProvider
    from app.ai.llm.openai_compatible import OpenAICompatibleLLM

    # Check ChromaDB
    vector_db_status = "unreachable"
    try:
        store = get_chroma_store()
        if await store.health_check():
            vector_db_status = "ok"
    except Exception:
        vector_db_status = "unreachable"

    # Check LLM
    llm_status = "unreachable"
    try:
        if settings.use_mock_llm:
            llm_status = "mock"
        else:
            llm = OpenAICompatibleLLM()
            if await llm.health_check():
                llm_status = "ok"
    except Exception:
        llm_status = "unreachable"

    # Determine overall status
    overall = "ok"
    if vector_db_status == "unreachable":
        overall = "degraded"

    return HealthStatus(
        status=overall,
        version=settings.app_version,
        uptime_seconds=round(time.time() - _START_TIME, 1),
        vector_db_status=vector_db_status,
        llm_status=llm_status,
        environment=settings.app_env,
    )
