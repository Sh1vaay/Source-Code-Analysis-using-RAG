"""
store_index.py
──────────────
CLI helper to clone a repository and pre-index it into Qdrant without
starting the API server. Useful for pre-loading a repo before deployment.

Usage:
    python store_index.py https://github.com/username/repo
"""
from __future__ import annotations

import asyncio
import sys

from src.ingestion import clone_repository, parse_repository
from src.retrieval import clear_collection, ensure_collection, index_chunks


async def main(repo_url: str) -> None:
    print(f"[store_index] Initialising collection...")
    await ensure_collection()

    print(f"[store_index] Clearing existing index...")
    await clear_collection()

    print(f"[store_index] Cloning {repo_url}...")
    repo_path = await clone_repository(repo_url)

    print(f"[store_index] Parsing repository...")
    chunks = await parse_repository(repo_path)
    if not chunks:
        print("[store_index] ERROR: No supported code files found in repository.", file=sys.stderr)
        sys.exit(1)

    print(f"[store_index] Indexing {len(chunks)} chunks into Qdrant...")
    count = await index_chunks(chunks)

    print(f"[store_index] Done. {count} chunks indexed successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python store_index.py <repo_url>", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))