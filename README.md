# ai_gen_platform

Production-grade GenAI backend with a full RAG pipeline, agent orchestrator, enterprise security, and Docker-first setup.

📐 **[Architecture & diagrams → ARCHITECTURE.md](ARCHITECTURE.md)** · Concepts → [CONCEPTS.md](CONCEPTS.md) · Guided code tour → [LEARNING_GUIDE.md](LEARNING_GUIDE.md)

A Postman collection with all endpoints is included: [`postman_collection.json`](postman_collection.json).

## Quick Start

```bash
cp .env.example .env
docker compose up --build
```

The API is now available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

**Default credentials:** `USE_MOCK_LLM=true` — no API key needed, no cost.

---

## All API Endpoints

### Health

```bash
# Liveness — is the app running?
curl http://localhost:8000/api/v1/health

# Readiness — are dependencies (ChromaDB, LLM) reachable?
curl http://localhost:8000/api/v1/health/ready
```

### Authentication

```bash
# Get a JWT token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secret"}'
# → {"access_token": "eyJ...", "token_type": "bearer", "expires_in": 3600}
```

### Chat (JWT required)

```bash
TOKEN="eyJ..."  # from /auth/token above

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is RAG?"}]
  }'
```

### RAG Query (API Key required)

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "X-API-Key: dev-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the security considerations for enterprise AI?"}'
```

### RAG Ingest (API Key required)

```bash
# Ingest raw text
curl -X POST http://localhost:8000/api/v1/rag/ingest \
  -H "X-API-Key: dev-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your document text here...",
    "metadata": {"source": "my_doc.md", "title": "My Document"}
  }'

# Ingest a file from the data/ directory
curl -X POST http://localhost:8000/api/v1/rag/ingest \
  -H "X-API-Key: dev-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "sample_docs/enterprise_ai_guide.md",
    "metadata": {"source": "enterprise_ai_guide.md", "title": "Enterprise AI Guide"}
  }'
```

### Agent (JWT required)

```bash
curl -X POST http://localhost:8000/api/v1/agents/run \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is chunking in RAG and why does it matter?"}'
```

```bash
# List available tools
curl http://localhost:8000/api/v1/agents/tools
```

---

## Makefile Commands

| Command | Description |
|---|---|
| `make dev` | Start all services with hot reload |
| `make test` | Run test suite inside Docker |
| `make ingest` | Ingest sample documents into ChromaDB |
| `make logs` | Tail application logs |
| `make shell` | Open bash in the app container |
| `make stop` | Stop all services |
| `make clean` | Stop and remove volumes (destroys ChromaDB data!) |
| `make rebuild` | Force rebuild all Docker images |
| `make seed` | Print sample JWT tokens and API key |

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `USE_MOCK_LLM` | `true` | Use mock LLM (no API calls) |
| `LLM_API_KEY` | `""` | OpenAI/compatible API key (only when USE_MOCK_LLM=false) |
| `LLM_MODEL` | `gpt-4o-mini` | Model name to use |
| `JWT_SECRET_KEY` | dev value | Secret for JWT signing (change in prod!) |
| `JWT_EXPIRE_MINUTES` | `60` | Token expiry in minutes |
| `API_KEY` | dev value | X-API-Key value for protected endpoints |
| `CHROMA_HOST` | `chroma` | ChromaDB hostname (use `chroma` in Docker) |
| `CHROMA_PORT` | `8001` | ChromaDB port |
| `CHUNK_SIZE` | `512` | Document chunk size in characters |
| `CHUNK_OVERLAP` | `64` | Chunk overlap in characters |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity to include a chunk |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model name |

---

## How to Add a New Document to the RAG Index

**Option 1: API (recommended for production)**
```bash
curl -X POST http://localhost:8000/api/v1/rag/ingest \
  -H "X-API-Key: your-api-key" \
  -d '{"content": "Document text...", "metadata": {"source": "doc.md"}}'
```

**Option 2: Drop a file and re-ingest**
1. Add your `.md` or `.txt` file to `data/sample_docs/`
2. Run `make ingest` (or `make rebuild-index` to wipe and rebuild)

---

## How to Add a New LLM Provider

1. Create a new class in `app/ai/llm/` that extends `BaseLLM`
2. Implement `complete()` and `stream()` methods
3. Update `app/api/v1/_providers.py` `get_llm()` to return your new class
4. Add any required env vars to `.env.example`

The rest of the codebase uses the `BaseLLM` interface — nothing else changes.
