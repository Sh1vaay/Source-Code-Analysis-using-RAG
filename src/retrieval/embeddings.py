"""
src/retrieval/embeddings.py
────────────────────────────
Embedding provider factory.

Supports three backends, switchable via EMBEDDING_PROVIDER in .env:
  • fastembed  — local ONNX model (BAAI/bge-small-en-v1.5), no API key, FREE
  • openai     — text-embedding-3-large, best general purpose, costs money
  • voyage     — voyage-code-2, best for code, free tier (50M tokens/month)

All return a callable embed(texts: list[str]) -> list[list[float]].
"""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from src.config import get_settings


class Embedder(Protocol):
    """Minimal embedding interface."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    @property
    def dimension(self) -> int: ...


# ── FastEmbed (local, free) ────────────────────────────────────────────────────
class FastEmbedEmbedder:
    MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, excellent for code
    _dimension = 384

    def __init__(self) -> None:
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=self.MODEL_NAME)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


# ── OpenAI embeddings ─────────────────────────────────────────────────────────
class OpenAIEmbedder:
    MODEL_NAME = "text-embedding-3-large"
    _dimension = 3072

    def __init__(self, api_key: str) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.MODEL_NAME, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


# ── Voyage AI embeddings (free tier — best for code) ─────────────────────────
class VoyageEmbedder:
    _dimension = 1536  # voyage-code-2 default

    def __init__(self, api_key: str, model: str) -> None:
        import voyageai
        self._client = voyageai.Client(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self._model, input_type="document")
        return result.embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self._client.embed([text], model=self._model, input_type="query")
        return result.embeddings[0]

    @property
    def dimension(self) -> int:
        return self._dimension


@lru_cache
def get_embedder() -> Embedder:
    """Return the singleton embedder configured by EMBEDDING_PROVIDER."""
    settings = get_settings()
    provider = settings.embedding_provider

    if provider == "fastembed":
        return FastEmbedEmbedder()
    elif provider == "openai":
        return OpenAIEmbedder(api_key=settings.openai_api_key)
    elif provider == "voyage":
        return VoyageEmbedder(api_key=settings.voyage_api_key, model=settings.voyage_model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider!r}")
