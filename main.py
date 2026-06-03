"""
main.py
────────
Top-level entrypoint kept for legacy compatibility and convenience.

Usage:
    python main.py                    # development server with reload
    uvicorn src.api.main:app ...      # production (preferred)
"""
import uvicorn

from src.api.main import app  # noqa: F401 — ensure app is importable from root
from src.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level="info",
    )