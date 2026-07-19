# Learning Guide: Guided Study Path

Work through this codebase in order. Each step builds on the previous one.

---

## Step 1: Understand the Services — `docker-compose.yml`

Start here to understand what runs and how it connects.

- Two services: `app` (FastAPI on port 8000) and `chroma` (ChromaDB on port 8001)
- The `app` service `depends_on: chroma: condition: service_healthy` — it waits for ChromaDB
- Named volumes: `chroma_data` persists the vector index, `model_cache` caches the embedding model
- Source code is mounted as a volume, enabling hot reload without rebuilding

**Run it:** `docker compose up --build`

---

## Step 2: Configuration — `app/core/config.py`

Learn how the app reads environment variables.

- `BaseSettings` from pydantic-settings reads `.env` files and environment variables
- Every configuration value has a type annotation — Pydantic validates and coerces automatically
- `@lru_cache` on `get_settings()` ensures settings are read once, not on every request
- `get_settings.cache_clear()` resets the cache in tests

**Key insight:** All configuration is in one place. No `os.getenv()` calls scattered in business logic.

---

## Step 3: Application Factory — `app/main.py`

Learn how the FastAPI app is assembled.

- `create_app()` is the factory function — it builds and returns the app
- `lifespan()` context manager handles startup (pre-warm embedding model) and shutdown
- Middleware, exception handlers, and rate limiting are registered before routes
- The `app = create_app()` at module level is what uvicorn imports

**Key insight:** The app factory pattern makes testing easy — call `create_app()` with test config.

---

## Step 4: Request Lifecycle — `app/core/middleware.py`

Understand what happens to every HTTP request.

- `RequestIDMiddleware`: assigns a UUID to every request, measures latency, logs it
- `SecurityHeadersMiddleware`: adds browser security headers to every response
- `CORSMiddleware`: handles browser preflight requests
- Middleware registration order = reverse execution order

**Key insight:** Middleware is a powerful pattern for cross-cutting concerns. Never put business logic in middleware.

---

## Step 5: The RAG Pipeline — `app/ai/rag/pipeline.py`

The most important AI file. Read every comment.

- Step 1: Input guardrails (injection check → PII masking → moderation)
- Step 2: Retrieval (`RAGRetriever.retrieve()` → embed query → vector search)
- Step 3: Context assembly (format retrieved chunks as `[Document N]\ntext`)
- Step 4: LLM generation (system prompt + context + question)
- Step 5: Cost tracking
- Step 6: Output validation
- Step 7: Output moderation
- Step 8: Citation building (append source list to answer)

Also read: `app/ai/rag/chunker.py` (how documents are split) and `app/ai/rag/retriever.py` (how queries are matched to chunks).

---

## Step 6: The Agent — `app/ai/agents/orchestrator.py`

Learn the single-agent pattern.

- The agent calls the LLM to decide which tool to use ("LLM-as-router")
- Then executes the chosen tool: retrieval, summarize, or direct answer
- Returns a step trace so you can see the reasoning
- `temperature=0.0` for the routing decision = deterministic tool selection

Compare this with the RAG pipeline: the pipeline is a fixed sequence, the agent is dynamic.

---

## Step 7: Run the Tests — `tests/`

Understand how mocking works.

```bash
make test
```

- `conftest.py`: `InMemoryVectorStore` replaces ChromaDB, `MockLLMProvider` replaces real LLM
- `test_client` fixture uses `ASGITransport` — no real HTTP server needed
- `auth_headers` fixture creates a real JWT token for test requests
- Run a single test file: `docker compose run --rm app pytest tests/test_guardrails.py -v`

---

## Step 8: Run Ingestion — `make ingest`

See real document ingestion.

```bash
make ingest
```

Watch the logs: embedding model loads, each document is chunked and embedded, vectors are stored in ChromaDB. After ingestion, you can query the knowledge base.

---

## Step 9: Query the RAG Endpoint

```bash
# Get a token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secret"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Run a RAG query
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "X-API-Key: dev-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is chunking in RAG and why does it matter?"}' | python3 -m json.tool
```

Notice: `answer` (LLM-generated), `sources` (retrieved chunks), `confidence_hint`, `retrieval_latency_ms`, `llm_latency_ms`.

---

## Step 10: Read the Architecture — `CONCEPTS.md`

Now that you've seen the code, read `CONCEPTS.md` for:
- ASCII diagrams of the RAG pipeline and agent loop
- How to swap ChromaDB for Azure AI Search
- How to swap mock LLM for OpenAI, Azure OpenAI, or Ollama
- Security layer diagram

---

## Key Patterns to Understand

| Pattern | Where to Find It | Why It Matters |
|---|---|---|
| Abstract base class / interface | `app/ai/llm/base.py`, `app/repositories/vector_store.py` | Decouples implementation from usage |
| Dependency injection | `app/core/dependencies.py`, `app/api/v1/_providers.py` | Makes components testable and swappable |
| Pydantic models for I/O | `app/models/` | Type safety at every layer |
| Structured logging | `app/core/logging.py` | Machine-parseable logs for observability |
| Guardrail pipeline | `app/ai/rag/pipeline.py` | Security layering |
| lru_cache singletons | `config.py`, `chroma_store.py`, `embeddings/provider.py` | Performance: load once, reuse |
| Factory function | `app/main.py:create_app()` | Testable app creation |
