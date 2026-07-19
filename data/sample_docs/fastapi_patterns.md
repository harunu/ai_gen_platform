# FastAPI Patterns: A Practical Developer Guide

**Platform Engineering | Developer Reference | Version 1.2**

---

## Dependency Injection in FastAPI

FastAPI's dependency injection system is one of its most powerful features. A dependency is any callable that FastAPI will call and inject into a route handler automatically.

**Basic pattern:**
```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/items")
async def list_items(db: Session = Depends(get_db)):
    return db.query(Item).all()
```

The `yield` keyword in a dependency makes it a context manager: code before `yield` runs on request, code after `yield` runs on response (even if an exception occurred). This is perfect for database connections, file handles, and HTTP clients.

**Sub-dependencies:** Dependencies can declare their own dependencies. FastAPI resolves the entire dependency tree automatically, sharing instances within a request (same `Depends` call = same instance).

**Overriding in tests:** Use `app.dependency_overrides` to replace real dependencies with mocks:
```python
app.dependency_overrides[get_db] = lambda: mock_db
```
This is cleaner than patching and works reliably with async dependencies.

---

## Async Endpoint Patterns and Common Mistakes

FastAPI supports both `async def` and regular `def` route handlers.

**Use `async def`** when your handler calls other async functions (database queries, HTTP calls, file I/O with async libraries). This is the correct choice for AI endpoints that call LLM APIs via `httpx.AsyncClient`.

**Use regular `def`** for CPU-bound work (image processing, heavy computation). FastAPI automatically runs regular `def` routes in a thread pool, keeping the event loop free.

**Common mistake — blocking the event loop:**
```python
# BAD: time.sleep blocks the entire event loop
@router.get("/slow")
async def slow_endpoint():
    time.sleep(5)  # blocks ALL other requests for 5 seconds!
    return {"status": "done"}

# GOOD: use asyncio.sleep for async waits
@router.get("/slow")
async def slow_endpoint():
    await asyncio.sleep(5)  # yields to event loop; other requests proceed
    return {"status": "done"}
```

**Common mistake — sync database calls in async routes:**
```python
# BAD: synchronous DB call inside async route
async def get_user(user_id: int, db=Depends(get_db)):
    return db.execute("SELECT ...")  # blocks event loop if db is sync

# GOOD: use async DB driver (asyncpg, aiomysql, motor) or run_in_executor
result = await loop.run_in_executor(None, lambda: db.execute("SELECT ..."))
```

---

## Middleware Design

Middleware intercepts every request before it reaches route handlers and every response before it is sent. Use middleware for:

- **Cross-cutting concerns** that apply to ALL endpoints: request ID injection, timing, CORS, security headers
- **Never** for business logic — that belongs in services

**Starlette BaseHTTPMiddleware** is the simplest way to write custom middleware:
```python
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time"] = str(time.perf_counter() - start)
        return response
```

**Middleware order matters:** Middleware is applied in reverse registration order. Register outermost middleware last. CORS middleware should be outermost (handles OPTIONS preflight before any auth runs).

---

## Background Tasks

FastAPI's `BackgroundTasks` runs a function after the response has been sent. Use it for work that doesn't need to complete before the user gets a response:

```python
@router.post("/ingest")
async def ingest(background_tasks: BackgroundTasks, body: IngestRequest):
    # Return immediately; run ingestion after response is sent
    background_tasks.add_task(run_ingestion, body.content)
    return {"status": "ingestion_queued"}
```

**Limitation:** Background tasks run in the same process as the API server. If the server restarts, queued tasks are lost. For durable background work, use Celery + Redis or a dedicated task queue.

---

## Testing FastAPI with pytest-asyncio

Testing async FastAPI endpoints requires an async HTTP client. Use `httpx.AsyncClient` with `ASGITransport`:

```python
@pytest.mark.asyncio
async def test_create_item(async_client: AsyncClient):
    response = await async_client.post("/items", json={"name": "widget"})
    assert response.status_code == 201

@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client
```

**Key points:**
- `ASGITransport` calls your app directly — no network, no port binding, fast
- Use `@pytest_asyncio.fixture` for async fixtures (not `@pytest.fixture`)
- Use `@pytest.mark.asyncio` for async test functions
- Set `asyncio_mode = "auto"` in `pytest.ini` to avoid marking every test manually

**Overriding dependencies in tests:**
```python
app.dependency_overrides[get_llm] = lambda: MockLLMProvider()
```
Always restore overrides after each test to avoid test pollution.
