"""
src/ingestion/parser.py
───────────────────────
Multi-language AST and sliding-window parser.

Performs:
  1. Exact Python AST parsing via Tree-sitter (extracts classes, functions, and methods).
  2. Generic sliding-window line chunking for other supported languages (JS/TS, Go, Rust, Java, C++, SQL, YAML, etc.).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Iterator

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

logger = logging.getLogger(__name__)

# Build the Python parser once at import time
_PY_LANGUAGE = Language(tspython.language())
_PARSER = Parser(_PY_LANGUAGE)

_EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".sh": "bash",
    ".bash": "bash",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".md": "markdown",
    ".txt": "text",
    ".html": "html",
    ".css": "css",
    ".ipynb": "jupyter",
}


@dataclass
class CodeChunk:
    """A single semantically extracted piece of source code."""

    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def page_content(self) -> str:  # LangChain-compatible alias
        return self.content


def _get_docstring(node: Node, source: bytes) -> str:
    """Return the docstring of a function/class node, or empty string."""
    body = next((c for c in node.children if c.type == "block"), None)
    if body is None:
        return ""
    first_stmt = next(iter(body.children), None)
    if first_stmt and first_stmt.type == "expression_statement":
        expr = next(iter(first_stmt.children), None)
        if expr and expr.type == "string":
            raw = source[expr.start_byte:expr.end_byte].decode("utf-8", errors="replace")
            return raw.strip('"""').strip("'''").strip()
    return ""


def _walk_chunks(node: Node, source: bytes, filepath: str) -> Iterator[CodeChunk]:
    """Recursively walk the AST and yield one CodeChunk per definition."""
    for child in node.children:
        if child.type in ("function_definition", "class_definition"):
            name_node = next((c for c in child.children if c.type == "identifier"), None)
            name = source[name_node.start_byte:name_node.end_byte].decode() if name_node else "unknown"
            content = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            docstring = _get_docstring(child, source)

            yield CodeChunk(
                content=content,
                metadata={
                    "source": filepath,
                    "name": name,
                    "type": child.type,          # "function_definition" | "class_definition"
                    "docstring": docstring,
                    "start_line": child.start_point[0] + 1,
                    "end_line": child.end_point[0] + 1,
                    "language": "python",
                },
            )
            # Recurse into class bodies to pick up methods
            yield from _walk_chunks(child, source, filepath)

        elif child.child_count > 0:
            yield from _walk_chunks(child, source, filepath)


def parse_python_file(path: Path, rel_path: str) -> list[CodeChunk]:
    """Parse a single Python file and return its AST-extracted chunks."""
    source = path.read_bytes()
    tree = _PARSER.parse(source)
    chunks = list(_walk_chunks(tree.root_node, source, rel_path))

    # Fallback: if no definitions found, treat the whole file as one chunk
    if not chunks:
        chunks = [
            CodeChunk(
                content=source.decode("utf-8", errors="replace"),
                metadata={
                    "source": rel_path,
                    "name": path.name,
                    "type": "module",
                    "docstring": "",
                    "start_line": 1,
                    "end_line": source.count(b"\n") + 1,
                    "language": "python",
                },
            )
        ]
    return chunks


def parse_generic_file(path: Path, rel_path: str, language: str) -> list[CodeChunk]:
    """Fallback chunker for non-python files: chunk by line sliding window."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    lines = content.splitlines()
    if not lines:
        return []

    # If the file is small, keep it as a single chunk
    if len(lines) <= 60:
        return [
            CodeChunk(
                content=content,
                metadata={
                    "source": rel_path,
                    "name": path.name,
                    "type": "file",
                    "docstring": "",
                    "start_line": 1,
                    "end_line": len(lines),
                    "language": language,
                },
            )
        ]

    # Otherwise, split into chunks of 50 lines with 10 lines overlap
    chunks = []
    chunk_size = 50
    overlap = 10
    step = chunk_size - overlap

    for i in range(0, len(lines), step):
        chunk_lines = lines[i : i + chunk_size]
        if not chunk_lines:
            break
        chunk_content = "\n".join(chunk_lines)
        chunks.append(
            CodeChunk(
                content=chunk_content,
                metadata={
                    "source": rel_path,
                    "name": f"{path.name} (lines {i+1}-{i+len(chunk_lines)})",
                    "type": "chunk",
                    "docstring": "",
                    "start_line": i + 1,
                    "end_line": i + len(chunk_lines),
                    "language": language,
                },
            )
        )
    return chunks


def parse_ipynb_file(path: Path, rel_path: str) -> list[CodeChunk]:
    """Parse a Jupyter Notebook file and return its code/markdown cells as chunks."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        notebook = json.loads(content)
    except Exception:
        return []

    cells = notebook.get("cells", [])
    if not cells:
        return []

    chunks = []
    cell_idx = 1
    for cell in cells:
        cell_type = cell.get("cell_type", "")
        source = cell.get("source", [])
        if isinstance(source, list):
            source_str = "".join(source)
        else:
            source_str = str(source)

        if not source_str.strip():
            continue

        chunks.append(
            CodeChunk(
                content=source_str,
                metadata={
                    "source": rel_path,
                    "name": f"{path.name} (cell {cell_idx})",
                    "type": cell_type,          # "code" | "markdown"
                    "docstring": "",
                    "start_line": cell_idx,
                    "end_line": cell_idx,
                    "language": "python" if cell_type == "code" else "markdown",
                },
            )
        )
        cell_idx += 1
    return chunks


# Maximum file size to parse (512 KB). Files larger than this are almost always
# generated/minified/binary-like and produce hundreds of useless chunks.
_MAX_FILE_BYTES = 512 * 1024  # 512 KB

# Directories to skip entirely during the repo walk
_SKIP_DIRS = {
    "__pycache__", "venv", "env", "node_modules",
    "dist", "build", "target", ".tox", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", "__snapshots__",
    "vendor", "third_party", "generated", ".gradle",
    "coverage", ".nyc_output",
}


def _sync_parse_repo(repo_path: Path) -> list[CodeChunk]:
    """Walk *repo_path* and parse every supported file found."""
    all_chunks: list[CodeChunk] = []
    logger.info("Starting repository parse at %s", repo_path)

    for path in sorted(repo_path.rglob("*")):
        if not path.is_file():
            continue

        # Compute path relative to repo root for dir-name checks
        try:
            rel_path = path.relative_to(repo_path)
        except ValueError:
            rel_path = path

        # Skip hidden dirs/files and known non-source directories
        parts = rel_path.parts
        if any(p.startswith(".") or p in _SKIP_DIRS for p in parts):
            continue

        ext = path.suffix.lower()
        if ext not in _EXTENSION_TO_LANGUAGE:
            continue

        # ── File size guard ───────────────────────────────────────────────────
        try:
            file_size = path.stat().st_size
        except OSError:
            continue

        if file_size > _MAX_FILE_BYTES:
            logger.warning(
                "Skipping %s: file too large (%d KB > %d KB limit)",
                rel_path, file_size // 1024, _MAX_FILE_BYTES // 1024,
            )
            continue

        if file_size == 0:
            continue  # empty file — nothing to index

        rel_path_str = str(rel_path)

        try:
            logger.debug("Parsing %s (%s)", rel_path, _EXTENSION_TO_LANGUAGE[ext])
            if ext == ".py":
                file_chunks = parse_python_file(path, rel_path_str)
            elif ext == ".ipynb":
                file_chunks = parse_ipynb_file(path, rel_path_str)
            else:
                file_chunks = parse_generic_file(path, rel_path_str, _EXTENSION_TO_LANGUAGE[ext])
            logger.debug("  -> %d chunks", len(file_chunks))
            all_chunks.extend(file_chunks)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", rel_path, e)

    logger.info("Repository parse complete. Total chunks: %d", len(all_chunks))
    return all_chunks



async def parse_repository(repo_path: Path) -> list[CodeChunk]:
    """Async wrapper around the blocking repo parse."""
    return await asyncio.to_thread(_sync_parse_repo, repo_path)
