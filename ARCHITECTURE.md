# Architecture

This document is the visual map of `ai_gen_platform` — a production-style GenAI backend
demonstrating a full RAG pipeline, an agent orchestrator, and enterprise security patterns
(JWT + API-key auth, rate limiting, guardrails, structured logging, cost telemetry).

For concept explanations see [CONCEPTS.md](CONCEPTS.md); for a guided tour of the code see
[LEARNING_GUIDE.md](LEARNING_GUIDE.md).

---

## System Overview

Two Docker services: the FastAPI app and ChromaDB (vector database). The LLM is either a
built-in mock (default, zero cost) or any OpenAI-compatible API.

```mermaid
flowchart LR
    client["Client<br/>(curl / Postman / frontend)"]

    subgraph docker["Docker (docker compose up --build)"]
        subgraph appsvc["app — FastAPI :8000"]
            mw["Middleware stack<br/>CORS → security headers → request ID<br/>+ rate limiting (slowapi)"]
            api["API v1 routes<br/>/health /auth /chat /rag /agents"]
            ai["AI layer<br/>RAG pipeline · Agent orchestrator · Guardrails"]
            emb["Embeddings<br/>sentence-transformers<br/>all-MiniLM-L6-v2 (local)"]
        end
        chroma[("chroma — ChromaDB :8001<br/>persistent vector store")]
    end

    llm["LLM provider<br/>MockLLM (default) or<br/>OpenAI-compatible API"]

    client -->|HTTP + JWT / X-API-Key| mw --> api --> ai
    ai --> emb
    ai <-->|similarity search / upsert| chroma
    ai <-->|complete / stream| llm
```

**Key properties**

| Concern | Implementation |
|---|---|
| Web framework | FastAPI (app factory + lifespan), hot reload in Docker |
| Vector store | ChromaDB service (FAISS available as an alternative in `app/repositories/`) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers, computed **locally** (no API cost) |
| LLM | `BaseLLM` interface → `MockProvider` (default) or `OpenAICompatibleLLM` |
| Auth | JWT bearer (chat, agents) and `X-API-Key` (RAG endpoints) |
| Safety | Prompt-injection detection, PII masking, moderation hooks, output validation |
| Observability | Structured logs with request IDs, per-request latency headers, LLM cost tracker |

---

## Request Flow: RAG Query

`POST /api/v1/rag/query` runs the full pipeline in `app/ai/rag/pipeline.py`:

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI route<br/>/api/v1/rag/query
    participant G as Guardrails
    participant R as Retriever
    participant V as ChromaDB
    participant L as LLM

    C->>API: query + X-API-Key
    API->>G: check_prompt_injection(query)
    G->>G: mask PII, moderate input
    G->>R: clean query
    R->>R: embed query (MiniLM, local)
    R->>V: similarity search (top_k, threshold)
    V-->>R: matching chunks + scores
    R-->>API: citations
    API->>L: system prompt + context block + question
    L-->>API: answer
    API->>G: validate output, moderate output
    API-->>C: answer + citations + confidence + latency breakdown
```

Steps in order: **guardrails → embed → retrieve → assemble context → LLM → cost tracking →
output validation → moderation → citations**. Retrieved context is injected into the *user*
message, not the system prompt, so the LLM treats it as information rather than instructions.

---

## Request Flow: Agent Run

`POST /api/v1/agents/run` uses an LLM-as-router pattern (`app/ai/agents/orchestrator.py`):
one LLM call decides the tool, then the tool executes. Simpler than ReAct, extendable to a
multi-step loop.

```mermaid
flowchart TD
    q["User query (JWT required)"] --> guard["Guardrails:<br/>injection check + PII masking"]
    guard --> mem["Load conversation memory<br/>(history from request)"]
    mem --> forced{"force_tool<br/>set?"}
    forced -->|yes| exec
    forced -->|no| route["LLM routing call (temp=0)<br/>returns JSON: {tool, reasoning}"]
    route -->|parse ok| exec{"Selected tool"}
    route -->|malformed JSON| fallback["Fallback:<br/>DIRECT_ANSWER"] --> exec
    exec -->|retrieval| rag["RAG pipeline<br/>(full flow above)"]
    exec -->|summarize| sum["Summarize tool<br/>(LLM call)"]
    exec -->|direct_answer| direct["Direct answer<br/>(LLM call)"]
    rag --> resp["Response:<br/>answer + step trace + latency"]
    sum --> resp
    direct --> resp
```

---

## Ingestion Flow

Documents enter the index via `POST /api/v1/rag/ingest` (raw text or a file under `data/`)
or `make ingest` (bulk-loads `data/sample_docs/`).

```mermaid
flowchart LR
    src["Document<br/>(text / .md / .txt)"] --> chunk["Chunker<br/>CHUNK_SIZE=512, OVERLAP=64"]
    chunk --> embed["Embed each chunk<br/>(MiniLM, local)"]
    embed --> store[("ChromaDB<br/>chunks + vectors + metadata")]
    store -.->|persisted in named volume<br/>chroma_data| disk["Survives restarts"]
```

---

## Module Map

```
app/
├── main.py                  # App factory + lifespan (pre-warms embedding model)
├── api/
│   ├── router.py            # Aggregates v1 routes under /api/v1
│   └── v1/                  # health, auth, chat, rag, agents endpoints
├── ai/                      # The AI layer — provider-agnostic
│   ├── llm/                 # BaseLLM interface, mock + OpenAI-compatible providers
│   ├── embeddings/          # Sentence-transformers embedding provider
│   ├── rag/                 # pipeline, retriever, chunker, citation builder
│   ├── agents/              # orchestrator (LLM-as-router), tools, memory
│   ├── guardrails/          # injection, PII, moderation, output validation
│   ├── prompts/             # All prompt templates in one place
│   ├── telemetry/           # LLM cost tracking
│   └── evaluation/          # RAG eval + prompt regression harnesses
├── core/                    # config, logging, security, middleware, rate limiting
├── models/                  # Pydantic request/response schemas
├── repositories/            # Vector store abstraction: Chroma + FAISS implementations
├── services/                # chat / document / ingestion services
└── utils/                   # ids, text, time helpers
```

**Swap points** (interfaces over implementations):

- **LLM** — implement `BaseLLM` in `app/ai/llm/`, wire it in `app/api/v1/_providers.py`.
- **Vector store** — `app/repositories/vector_store.py` defines the interface; Chroma and
  FAISS both implement it.
- **Moderation** — `app/ai/guardrails/moderation.py` is a stub ready for a real moderation API.
