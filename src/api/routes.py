"""
src/api/routes.py
──────────────────
FastAPI router definitions.

Endpoints:
  POST /api/ingest          — Clone a repo and index it into Qdrant
  GET  /api/chat            — SSE streaming chat (via query param)
  POST /api/chat            — JSON chat (non-streaming, for testing)
  GET  /api/health          — Health check
  DELETE /api/collection    — Clear the current vector collection
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl, field_validator

from src.chains.graph import run_rag, stream_rag
from src.config import get_settings
from src.ingestion import clone_repository, parse_repository
from src.retrieval import clear_collection, index_chunks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["RAG"])


# ── Request / response models ─────────────────────────────────────────────────
class IngestRequest(BaseModel):
    repo_url: str

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("repo_url must not be empty")
        if len(v) > 512:
            raise ValueError("repo_url is too long (max 512 chars)")
        return v


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int
    repo_url: str


class ChatRequest(BaseModel):
    question: str
    chat_history: list[dict] = []

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        if len(v) > 2000:
            raise ValueError("question is too long (max 2000 chars)")
        return v


class ChatResponse(BaseModel):
    answer: str



# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Simple liveness probe."""
    settings = get_settings()
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
        "qdrant_url": settings.qdrant_url,
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest_repository(payload: IngestRequest, background_tasks: BackgroundTasks):
    """
    Clone *repo_url* and index it into Qdrant.

    Clones the repository, runs AST parsing on Python files, and indexes them.
    The response returns immediately once indexing is complete.
    """
    settings = get_settings()

    try:
        # 1. Wipe any previous index
        await clear_collection()

        # 2. Clone the repo (async, runs in thread pool)
        repo_path = await clone_repository(payload.repo_url)

        # 3. Parse with Tree-sitter
        chunks = await parse_repository(repo_path)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No supported code files found in this repository.",
            )

        # 4. Embed + store in Qdrant
        count = await index_chunks(chunks)

        # 5. Schedule cleanup of the cloned repo to save disk space
        def _cleanup():
            shutil.rmtree(repo_path, ignore_errors=True)

        background_tasks.add_task(_cleanup)

        return IngestResponse(
            message=f"Successfully indexed {count} code chunks.",
            chunks_indexed=count,
            repo_url=payload.repo_url,
        )

    except ValueError as e:
        # URL validation failures
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e


@router.get("/chat")
async def chat_stream(
    question: str = Query(
        ...,
        description="The question to ask about the codebase",
        max_length=2000,
    ),
):
    """
    SSE streaming endpoint — returns tokens as they are generated.

    The browser / client should consume this as an EventSource.
    Events:
      \u2022 event: status — pipeline step updates ("Searching codebase...", etc.)
      \u2022 event: token  — one LLM token (JSON-encoded to handle newlines safely)
      \u2022 event: done   — signals end of stream
    """
    async def event_generator() -> AsyncIterator[str]:
        try:
            async for chunk in stream_rag(question):
                yield chunk
        except Exception as e:
            logger.exception("Error during SSE streaming for question: %s", question[:80])
            # Return a generic error to the client — do not expose internal details
            yield "event: error\ndata: An internal error occurred. Please try again.\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_json(payload: ChatRequest):
    """
    Non-streaming JSON endpoint — waits for the full answer.
    Useful for testing, scripting, or clients that don't support SSE.
    """
    try:
        answer = await run_rag(payload.question, payload.chat_history)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/collection")
async def delete_collection():
    """Clear the Qdrant collection (useful when switching repos)."""
    await clear_collection()
    return {"message": "Collection cleared successfully."}


@router.get("/debug")
async def debug_collection():
    """Get the current count of vectors in Qdrant."""
    from src.retrieval.vector_store import get_qdrant_client
    client = get_qdrant_client()
    settings = get_settings()
    try:
        if not client.collection_exists(settings.qdrant_collection):
            return {"exists": False, "count": 0}
        info = client.get_collection(settings.qdrant_collection)
        return {
            "exists": True,
            "count": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {"error": str(e)}
