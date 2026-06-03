"""
src/retrieval/vector_store.py
──────────────────────────────
Qdrant vector store manager.
"""
from __future__ import annotations

import asyncio
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import get_settings
from src.retrieval.embeddings import get_embedder

logger = logging.getLogger(__name__)

_client = None


def get_qdrant_client() -> QdrantClient:
    """Return a singleton instance of QdrantClient."""
    global _client
    if _client is None:
        settings = get_settings()
        if settings.qdrant_url == ":memory:":
            _client = QdrantClient(":memory:")
        else:
            # Pass api_key for authenticated remote Qdrant instances
            _client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
    return _client


async def ensure_collection():
    """Ensure the Qdrant collection exists and matches the embedder dimension."""
    client = get_qdrant_client()
    settings = get_settings()
    embedder = get_embedder()

    def _run():
        if not client.collection_exists(settings.qdrant_collection):
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=embedder.dimension,
                    distance=Distance.COSINE,
                ),
            )

    await asyncio.to_thread(_run)


async def clear_collection():
    """Wipe and re-create the Qdrant collection."""
    client = get_qdrant_client()
    settings = get_settings()
    embedder = get_embedder()

    def _run():
        if client.collection_exists(settings.qdrant_collection):
            client.delete_collection(settings.qdrant_collection)
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=embedder.dimension,
                distance=Distance.COSINE,
            ),
        )

    await asyncio.to_thread(_run)


async def index_chunks(chunks: list, batch_size: int = 64) -> int:
    """
    Embed code chunks and upsert them to Qdrant in small batches.

    Processing in batches prevents OOM crashes on large repositories by
    ensuring only `batch_size` chunks are held in memory at any one time.

    Parameters
    ----------
    chunks     : All CodeChunk objects from the parser.
    batch_size : Number of chunks to embed + upsert per iteration.
                 64 is a safe default for 8 GB RAM machines.
    """
    if not chunks:
        return 0
    client = get_qdrant_client()
    settings = get_settings()
    embedder = get_embedder()
    total_indexed = 0

    # Use batch size from settings, falling back to the parameter default
    effective_batch_size = settings.indexing_batch_size if batch_size == 64 else batch_size

    logger.info("Indexing %d chunks in batches of %d...", len(chunks), effective_batch_size)

    for batch_start in range(0, len(chunks), effective_batch_size):
        batch = chunks[batch_start : batch_start + effective_batch_size]
        texts = [chunk.content for chunk in batch]

        # Embed this batch (CPU/network-bound — run in thread pool)
        embeddings = await asyncio.to_thread(embedder.embed, texts)

        def _upsert(batch=batch, embeddings=embeddings) -> int:
            points = [
                PointStruct(
                    # UUID strings avoid ID collisions on incremental ingestion
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "content": chunk.content,
                        **chunk.metadata,
                    },
                )
                for chunk, vector in zip(batch, embeddings)
            ]
            client.upsert(
                collection_name=settings.qdrant_collection,
                points=points,
            )
            return len(points)

        count = await asyncio.to_thread(_upsert)
        total_indexed += count
        logger.info(
            "  Indexed batch %d/%d (%d chunks so far)",
            batch_start // effective_batch_size + 1,
            (len(chunks) + effective_batch_size - 1) // effective_batch_size,
            total_indexed,
        )

    logger.info("Indexing complete. Total chunks stored: %d", total_indexed)
    return total_indexed



async def similarity_search(query: str, top_k: int = 5) -> list[dict]:
    """Search Qdrant for matching documents via dense vector similarity."""
    client = get_qdrant_client()
    settings = get_settings()
    embedder = get_embedder()

    query_vector = await asyncio.to_thread(embedder.embed_query, query)

    def _run():
        results = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=top_k,
        )
        docs = []
        for hit in results:
            payload = hit.payload or {}
            docs.append({
                "content": payload.get("content", ""),
                "source": payload.get("source", "unknown"),
                "name": payload.get("name", ""),
                "type": payload.get("type", "module"),
                "docstring": payload.get("docstring", ""),
                "start_line": payload.get("start_line", 1),
                "end_line": payload.get("end_line", 1),
                "language": payload.get("language", "unknown"),
                "score": hit.score,
            })
        return docs

    return await asyncio.to_thread(_run)


async def filename_search(filename_hint: str, limit: int = 30) -> list[dict]:
    """
    Fetch all stored chunks whose `source` path contains *filename_hint*.

    This is used when the user's question explicitly mentions a file name or
    path segment (e.g. "gpt_instruction_finetuning.py"). Embedding similarity
    often misses these because the file path is not in the chunk text itself.

    Uses Qdrant scroll with a payload filter — no embedding needed.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchText

    client = get_qdrant_client()
    settings = get_settings()

    def _run() -> list[dict]:
        # MatchText does a case-insensitive substring match on the payload field
        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchText(text=filename_hint),
                )
            ]
        )
        records, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        docs = []
        for record in records:
            payload = record.payload or {}
            docs.append({
                "content": payload.get("content", ""),
                "source": payload.get("source", "unknown"),
                "name": payload.get("name", ""),
                "type": payload.get("type", "module"),
                "docstring": payload.get("docstring", ""),
                "start_line": payload.get("start_line", 1),
                "end_line": payload.get("end_line", 1),
                "language": payload.get("language", "unknown"),
                "score": 1.0,  # exact filename match — treat as maximum relevance
            })
        # Sort by line number so chunks are read in order
        docs.sort(key=lambda d: d["start_line"])
        return docs

    results = await asyncio.to_thread(_run)
    if results:
        logger.info(
            "Filename search for %r: found %d chunks", filename_hint, len(results)
        )
    return results

