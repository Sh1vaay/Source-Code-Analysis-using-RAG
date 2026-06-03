"""
src/api/main.py
────────────────
FastAPI application entry-point.

Provides:
  • async lifespan startup (initialise Qdrant collection)
  • CORS middleware for frontend dev servers
  • Static file serving for the frontend
  • LangSmith tracing activated when LANGCHAIN_TRACING_V2=true
  • Automatic OpenAPI docs at /docs
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.config import get_settings
from src.retrieval.vector_store import ensure_collection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before serving requests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    settings = get_settings()

    # Configure LangSmith tracing if enabled
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    # Ensure Qdrant collection exists
    await ensure_collection()

    logger.info("Source Code Analysis API is ready.")
    logger.info("  LLM provider   : %s", settings.llm_provider)
    logger.info("  Embeddings     : %s", settings.embedding_provider)
    logger.info("  Vector store   : %s", settings.qdrant_url)
    logger.info("  LangSmith      : %s", 'enabled' if settings.langchain_tracing_v2 else 'disabled')

    yield  # Application runs here

    logger.info("Shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Source Code Analysis using GenAI",
        description=(
            "A production-grade RAG system for interactively querying any "
            "GitHub repository using LangGraph + Qdrant + local cross-encoder reranking."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    # When credentials are required, wildcard origins are disallowed by spec.
    # Use CORS_ORIGINS env var (comma-separated) to configure allowed origins.
    # In development with no credentials needed, "*" works fine.
    cors_origins = settings.cors_origins_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,  # Set True only if you restrict origins
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # ── API routes ───────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Static files ─────────────────────────────────────────────────────────
    import pathlib
    static_dir = pathlib.Path(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ── Serve the SPA index.html for all non-API routes ──────────────────────
    templates_dir = pathlib.Path(__file__).parent.parent.parent / "templates"
    index_html = templates_dir / "index.html"

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(index_html))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level="info",
    )
