# WHY THIS FILE EXISTS:
# Tests for the /api/v1/health endpoints.
# Health checks are the first thing that breaks in a deployment — test them thoroughly.

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_liveness_returns_200(test_client: AsyncClient) -> None:
    """Liveness probe must return 200 with status='ok'."""
    response = await test_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_liveness_returns_version(test_client: AsyncClient) -> None:
    """Liveness probe must include the app version."""
    response = await test_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert len(data["version"]) > 0


@pytest.mark.asyncio
async def test_liveness_returns_uptime(test_client: AsyncClient) -> None:
    """Liveness probe must include uptime_seconds."""
    response = await test_client.get("/api/v1/health")
    data = response.json()
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_readiness_returns_200(test_client: AsyncClient) -> None:
    """
    Readiness probe must return 200 even when ChromaDB is the in-memory mock.
    The status may be 'degraded' if ChromaDB is unreachable, but HTTP must be 200.
    """
    response = await test_client.get("/api/v1/health/ready")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_readiness_mock_llm_status(test_client: AsyncClient) -> None:
    """With USE_MOCK_LLM=true, the readiness probe should report llm_status='mock'."""
    from app.core.config import get_settings
    settings = get_settings()
    if settings.use_mock_llm:
        response = await test_client.get("/api/v1/health/ready")
        data = response.json()
        assert data["llm_status"] == "mock"


@pytest.mark.asyncio
async def test_response_includes_request_id_header(test_client: AsyncClient) -> None:
    """Every response must include X-Request-ID for request tracing."""
    response = await test_client.get("/api/v1/health")
    assert "x-request-id" in response.headers


@pytest.mark.asyncio
async def test_response_includes_timing_header(test_client: AsyncClient) -> None:
    """Every response must include X-Response-Time-Ms."""
    response = await test_client.get("/api/v1/health")
    assert "x-response-time-ms" in response.headers
