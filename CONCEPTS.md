# Architectural Concepts

## Why Clean Architecture?

This project separates concerns into layers:

```
HTTP (routes) → Services (business logic) → AI (LLM/RAG) → Repositories (storage)
```

Each layer only knows about the layer directly below it. Routes don't touch ChromaDB. Services don't parse HTTP. This makes each layer independently testable and replaceable.

---

## RAG Pipeline Flow

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  Guardrails (Input)             │
│  ├─ Prompt injection check      │
│  ├─ PII masking                 │
│  └─ Content moderation          │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Embedding                      │
│  Query → 384-dim vector         │
│  (sentence-transformers)        │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Vector Search (ChromaDB)       │
│  Cosine similarity → top-K      │
│  chunks + metadata filter       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Context Assembly               │
│  [Doc 1 - Title]\ntext\n\n      │
│  [Doc 2 - Title]\ntext          │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  LLM Generation                 │
│  System: answer from context    │
│  User: context + question       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Guardrails (Output)            │
│  ├─ Output validation           │
│  └─ Output moderation           │
└─────────────────────────────────┘
    │
    ▼
Response: answer + sources + confidence + latency
```

---

## Agent Decision Loop

```
User Query
    │
    ▼
┌──────────────────────────────────┐
│  Guardrails + PII masking        │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│  LLM Router                      │
│  "Which tool should handle this?"│
│  → RETRIEVAL | SUMMARIZE         │
│  → DIRECT_ANSWER                 │
└──────────────────────────────────┘
    │
    ├─ RETRIEVAL ──→ RAG Pipeline → answer
    │
    ├─ SUMMARIZE ──→ LLM(summarize prompt) → summary
    │
    └─ DIRECT   ──→ LLM(direct prompt) → answer
    │
    ▼
Response: answer + step trace + tool used + latency
```

The step trace (`steps[]` in the response) shows exactly which tool was selected and why — essential for debugging agent behavior.

---

## Security Layers

```
Internet
    │
    ▼
┌─────────────────────────────────┐
│  CORS Middleware                │
│  Blocks cross-origin requests   │
│  from unlisted origins          │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Rate Limiting (slowapi)        │
│  10 req/min on AI endpoints     │
│  60 req/min on health           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Auth (JWT / API Key)           │
│  FastAPI dependency injection   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Input Guardrails               │
│  ├─ Prompt injection detection  │
│  ├─ PII masking                 │
│  └─ Content moderation hook     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Business Logic + LLM           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Output Guardrails              │
│  ├─ Output validation           │
│  └─ Output moderation hook      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Security Headers Middleware    │
│  X-Content-Type-Options: nosniff│
│  X-Frame-Options: DENY          │
└─────────────────────────────────┘
    │
    ▼
Client
```

---

## How to Swap ChromaDB for Azure AI Search

1. Create `app/repositories/azure_ai_search_store.py` implementing `VectorStore`
2. Implement `upsert()`, `query()`, `delete()`, `count()`, `health_check()`
   - Azure AI Search uses REST API or the `azure-search-documents` SDK
   - Map `where` filter to OData filter expressions
3. In `app/repositories/chroma_store.py`, replace `get_chroma_store()` with `get_azure_store()`
4. Update `docker-compose.yml` to remove the `chroma` service
5. Update `.env.example` with Azure Search endpoint and key

**Nothing else in the codebase changes.** This is why the abstract `VectorStore` interface exists.

Azure Migration Path:
- Vector Store: ChromaDB → Azure AI Search (Vector)
- App Hosting: Docker → Azure Container Apps
- LLM: Mock/OpenAI → Azure OpenAI (update `LLM_BASE_URL` to Azure endpoint)
- Auth: JWT → integrate with Azure AD via `msal` library

---

## How to Swap Mock LLM for OpenAI or Azure OpenAI

**For OpenAI:**
```bash
# In .env:
USE_MOCK_LLM=false
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-key
LLM_MODEL=gpt-4o-mini
```

**For Azure OpenAI:**
```bash
USE_MOCK_LLM=false
LLM_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
LLM_API_KEY=your-azure-key
LLM_MODEL=gpt-4o  # deployment name
```

Azure OpenAI uses the same `/chat/completions` API path — `OpenAICompatibleLLM` works without changes.

**For Ollama (local open-source models):**
```bash
USE_MOCK_LLM=false
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama  # arbitrary string
LLM_MODEL=llama3
```
