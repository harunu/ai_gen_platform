# WHY THIS FILE EXISTS:
# Centralizes all application configuration in one place using Pydantic Settings.
# Pydantic Settings reads values from environment variables and .env files,
# validates their types, and exposes them as a typed Python object.
# This pattern prevents scattered os.getenv() calls throughout the codebase.

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Pydantic Settings works like a regular Pydantic model, but it also reads
    values from:
      1. Environment variables (highest priority)
      2. .env file (if env_file is configured)
      3. Default values defined here (lowest priority)

    Type annotations drive automatic coercion:
      - "true" / "false" strings → bool
      - "8000" → int
      - etc.
    """

    # Pydantic Settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",            # Read from .env file if present
        env_file_encoding="utf-8",
        case_sensitive=False,       # APP_NAME and app_name are treated the same
        extra="ignore",             # Ignore unknown env vars silently
    )

    # ── Application ──────────────────────────────────────────
    app_name: str = Field(default="ai_gen_platform")
    app_version: str = Field(default="1.0.0")
    app_env: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # ── Server ───────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # ── Security ─────────────────────────────────────────────
    jwt_secret_key: str = Field(default="dev-secret-change-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=60)
    api_key: str = Field(default="dev-api-key-change-in-production")

    # CORS: stored as a string in env, split into a list here
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:8080")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str) -> str:
        # Keep as string; we'll split when needed
        return v

    def get_cors_origins(self) -> List[str]:
        """Return CORS origins as a list of strings."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # ── LLM ─────────────────────────────────────────────────
    use_mock_llm: bool = Field(default=True)   # True = no API calls, safe for local dev
    llm_base_url: str = Field(default="https://api.openai.com/v1")
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_timeout_seconds: int = Field(default=30)
    llm_max_tokens: int = Field(default=2048)
    llm_temperature: float = Field(default=0.7)

    # ── Embeddings ───────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_batch_size: int = Field(default=32)

    # ── ChromaDB ─────────────────────────────────────────────
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8001)
    chroma_collection_name: str = Field(default="documents")

    # ── RAG Pipeline ─────────────────────────────────────────
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    top_k: int = Field(default=5)
    similarity_threshold: float = Field(default=0.3)

    # ── Rate Limiting ────────────────────────────────────────
    rate_limit_ai: str = Field(default="10/minute")
    rate_limit_health: str = Field(default="60/minute")

    # ── Agent ────────────────────────────────────────────────
    agent_max_steps: int = Field(default=5)
    agent_memory_turns: int = Field(default=10)

    @property
    def chroma_url(self) -> str:
        """Convenience property: full ChromaDB HTTP URL."""
        return f"http://{self.chroma_host}:{self.chroma_port}"

    @property
    def is_production(self) -> bool:
        """True when running in production environment."""
        return self.app_env.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    lru_cache(maxsize=1) means this function is called once and the result
    is reused for the lifetime of the process. This prevents re-reading the
    .env file on every request.

    In tests, call get_settings.cache_clear() to reset and inject test values.
    """
    return Settings()
