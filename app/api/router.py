# WHY THIS FILE EXISTS:
# Aggregates all v1 sub-routers into a single APIRouter that main.py includes.
# Versioning all routes under /api/v1 means we can add /api/v2 later without
# breaking existing clients.

from fastapi import APIRouter

from app.api.v1 import agents, auth, chat, health, rag

# The top-level API router — all versioned routes live under /api
api_router = APIRouter(prefix="/api")

# v1 routes — each sub-router defines its own prefix:
#   health  → /api/v1/health
#   auth    → /api/v1/auth
#   chat    → /api/v1/chat
#   rag     → /api/v1/rag
#   agents  → /api/v1/agents
v1_router = APIRouter(prefix="/v1")

v1_router.include_router(health.router)
v1_router.include_router(auth.router)
v1_router.include_router(chat.router)
v1_router.include_router(rag.router)
v1_router.include_router(agents.router)

api_router.include_router(v1_router)
