"""
src/config.py
─────────────
Central configuration loaded from environment variables via pydantic-settings.
All modules import from here — no scattered os.environ calls.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM provider ─────────────────────────────────────────────────────────
    llm_provider: Literal["groq", "openai", "ollama"] = "groq"

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_provider: Literal["fastembed", "openai", "voyage"] = "fastembed"
    voyage_api_key: str = ""
    voyage_model: str = "voyage-code-2"

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_url: str = ":memory:"
    qdrant_api_key: str = ""
    qdrant_collection: str = "code_rag"

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    chunk_size: int = 1500
    chunk_overlap: int = 200
    # Number of chunks to embed + upsert per batch. Lower = less RAM used.
    indexing_batch_size: int = 64
    # Skip files larger than this (KB). Prevents OOM on minified/generated files.
    max_file_kb: int = 512

    # ── Observability ────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "code-rag"

    # ── App ──────────────────────────────────────────────────────────────────
    app_host: str = "127.0.0.1"
    app_port: int = 8080
    app_debug: bool = False
    repo_clone_dir: str = "./repo"
    allowed_hosts: str = "github.com,gitlab.com,bitbucket.org"
    # Comma-separated allowed CORS origins. Use "*" for open dev access.
    cors_origins: str = "*"

    @property
    def allowed_hosts_list(self) -> list[str]:
        return [h.strip() for h in self.allowed_hosts.split(",")]

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _validate_provider_keys(self) -> "Settings":
        """Fail fast at startup if the configured provider has no API key."""
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ValueError(
                "LLM_PROVIDER=groq requires GROQ_API_KEY to be set in .env"
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER=openai requires OPENAI_API_KEY to be set in .env"
            )
        if self.embedding_provider == "voyage" and not self.voyage_api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=voyage requires VOYAGE_API_KEY to be set in .env"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton Settings object."""
    return Settings()
