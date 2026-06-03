"""
src/retrieval/reranker.py
──────────────────────────
Cross-encoder reranker using a local SentenceTransformers model.

Stage 1: Qdrant dense search → top-K candidates  (vector_store.py)
Stage 2: Cross-encoder reranks top-K → returns top-N (this file)

The cross-encoder reads the full query + each document pair and outputs a
relevance score that is far more accurate than dot-product similarity.
No API calls — runs entirely on CPU with a small (22M param) model.
"""
from __future__ import annotations

from functools import lru_cache

from src.config import get_settings

# Model that's small enough to run on CPU in < 200 ms per batch
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache
def _get_cross_encoder():
    """Lazily load the cross-encoder (downloaded once, cached in memory)."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(_CROSS_ENCODER_MODEL)


def rerank(query: str, documents: list[dict], top_n: int | None = None) -> list[dict]:
    """
    Rerank *documents* for *query* using a cross-encoder.

    Parameters
    ----------
    query     : The user question.
    documents : List of dicts, each must have a ``content`` key.
    top_n     : Number of results to return after reranking.

    Returns
    -------
    List of dicts sorted by cross-encoder relevance (highest first),
    each augmented with a ``rerank_score`` field.
    """
    if not documents:
        return []

    settings = get_settings()
    n = top_n or settings.rerank_top_n

    cross_encoder = _get_cross_encoder()
    pairs = [(query, doc["content"]) for doc in documents]
    scores = cross_encoder.predict(pairs)  # numpy array

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )

    result = []
    for score, doc in ranked[:n]:
        result.append({**doc, "rerank_score": float(score)})

    return result
