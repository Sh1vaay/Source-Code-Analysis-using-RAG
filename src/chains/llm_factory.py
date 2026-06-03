"""
src/chains/llm_factory.py
──────────────────────────
LLM provider factory.

Supports three backends, switchable via LLM_PROVIDER in .env:
  • groq   — Groq API (free tier): Llama-3-8b, Mixtral-8x7b, Gemma-2-9b
  • ollama — Fully local execution via Ollama server, 100% free
  • openai — OpenAI API, optional paid fallback

All return a LangChain-compatible chat model (BaseChatModel).
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from src.config import get_settings


@lru_cache
def get_llm() -> BaseChatModel:
    """Return the singleton LLM configured by LLM_PROVIDER."""
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=0.2,
            streaming=True,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.2,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0.2,
            streaming=True,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            "Set LLM_PROVIDER=groq|ollama|openai in your .env file."
        )
