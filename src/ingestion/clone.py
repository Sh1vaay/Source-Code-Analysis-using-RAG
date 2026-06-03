"""
src/ingestion/clone.py
──────────────────────
Secure repository cloning using pygit2 (libgit2 bindings).

Includes:
  • URL allowlisting to prevent SSRF
  • Validation that the URL resolves to a known git host
  • Asynchronous clone (runs in a thread pool to avoid blocking FastAPI)
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from urllib.parse import urlparse

import pygit2

from src.config import get_settings


def _validate_repo_url(url: str) -> None:
    """Raise ValueError if the URL is not from an allowed host."""
    settings = get_settings()
    parsed = urlparse(url)

    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"Only http/https URLs are allowed. Got: {parsed.scheme!r}")

    host = parsed.netloc.lower().split(":")[0]  # strip port
    if not any(host == allowed or host.endswith(f".{allowed}")
               for allowed in settings.allowed_hosts_list):
        raise ValueError(
            f"Host '{host}' is not in the allowlist: {settings.allowed_hosts_list}"
        )


def _sync_clone(url: str, dest: Path) -> Path:
    """Blocking clone — intended to run inside asyncio.to_thread()."""
    _validate_repo_url(url)

    # Remove any stale clone directory
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    pygit2.clone_repository(url, str(dest))
    return dest


async def clone_repository(url: str, dest: Path | None = None) -> Path:
    """
    Async wrapper — clones *url* into *dest* without blocking the event loop.

    Parameters
    ----------
    url : str
        Public HTTPS git repository URL.
    dest : Path | None
        Target directory. Defaults to ``settings.repo_clone_dir / <repo_name>``.

    Returns
    -------
    Path
        The directory the repo was cloned into.
    """
    settings = get_settings()

    if dest is None:
        repo_name = urlparse(url).path.rstrip("/").split("/")[-1]
        repo_name = repo_name.removesuffix(".git")
        # Sanitize to prevent directory traversal (e.g. "../") or invalid file names
        repo_name = "".join(c for c in repo_name if c.isalnum() or c in ("-", "_"))
        if not repo_name:
            repo_name = "default_repo"
        dest = Path(settings.repo_clone_dir) / repo_name

    return await asyncio.to_thread(_sync_clone, url, dest)
