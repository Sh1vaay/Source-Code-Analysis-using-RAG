"""
src/chains/graph.py
────────────────────
LangGraph RAG state machine.

The graph has four nodes executed in sequence:
  1. query_transform  — HyDE: ask LLM to write a hypothetical code snippet,
                        then embed *that* for retrieval (improves recall)
  2. retrieve         — hybrid search:
                          a) dense Qdrant similarity search (semantic)
                          b) payload-filtered filename search (when a file is named)
                        results are merged and deduplicated before reranking
  3. rerank           — local cross-encoder narrows to top-N
  4. generate         — LLM synthesises answer from reranked context

This is fully observable: every node's input/output is captured if
LANGCHAIN_TRACING_V2=true in .env (LangSmith).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncIterator, TypedDict

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from src.chains.llm_factory import get_llm
from src.config import get_settings
from src.retrieval.reranker import rerank
from src.retrieval.vector_store import filename_search, similarity_search

logger = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────
class RagState(TypedDict):
    """State passed between LangGraph nodes."""
    question: str
    chat_history: list[dict]          # [{"role": "user"|"assistant", "content": str}]
    transformed_query: str
    filename_hints: list[str]         # file names/paths detected in the question
    raw_docs: list[dict]
    reranked_docs: list[dict]
    answer: str


# ── Filename hint extraction ───────────────────────────────────────────────────
# Matches things like: foo.py  /path/to/bar.js  ch07/02_dataset-utilities
_FILENAME_PATTERN = re.compile(
    r"""
    (?:
        # Full or partial paths with slashes
        (?:[\w\-\.]+/)+[\w\-\.]+
        |
        # Standalone filenames with known code extensions
        [\w\-\.]+\.(?:py|js|ts|jsx|tsx|go|rs|java|kt|cs|cpp|c|rb|sh|sql|yaml|yml|json|toml|md|ipynb)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def extract_filename_hints(question: str) -> list[str]:
    """
    Extract file names and path segments mentioned in the question.

    Returns a deduplicated list of strings to use as filename search hints.
    Each hint will be matched against the ``source`` field in Qdrant via
    case-insensitive substring match.

    Examples
    --------
    >>> extract_filename_hints("what does gpt_instruction_finetuning.py do?")
    ['gpt_instruction_finetuning.py']
    >>> extract_filename_hints("explain ch07/02_dataset-utilities")
    ['ch07/02_dataset-utilities']
    """
    matches = _FILENAME_PATTERN.findall(question)
    # Deduplicate while preserving order
    seen: set[str] = set()
    hints: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            hints.append(m)
    return hints


# ── Prompt templates ──────────────────────────────────────────────────────────
_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are an expert software engineer. "
        "Given a question about a codebase, write a SHORT hypothetical "
        "code snippet or explanation that would answer it. "
        "This is used for retrieval, not for the final answer. "
        "Be concise (max 10 lines)."
    )),
    ("human", "{question}"),
])

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are an expert software engineer helping a developer understand a codebase.\n"
        "You will be given excerpts from source files with their file paths and line numbers.\n"
        "Answer the question using ONLY the provided code context.\n"
        "Always cite the file path when referring to specific code.\n"
        "If the context doesn't contain enough information, say so clearly.\n"
        "Format code with markdown code blocks.\n"
        "Be precise and concise."
    )),
    ("human", (
        "Code Context:\n"
        "─────────────\n"
        "{context}\n"
        "─────────────\n\n"
        "Question: {question}"
    )),
])


def _build_context(docs: list[dict]) -> str:
    """Format retrieved docs into a context string with clear source attribution."""
    parts = []
    for doc in docs:
        source = doc.get("source", "unknown")
        name = doc.get("name", "")
        start = doc.get("start_line", "?")
        end = doc.get("end_line", "?")
        lang = doc.get("language", "")

        # Header: file path + line range + symbol name if available
        header = f"### File: {source}  (lines {start}–{end})"
        if name and name != source:
            header += f"  |  symbol: `{name}`"

        fence = f"```{lang}" if lang else "```"
        parts.append(f"{header}\n{fence}\n{doc['content']}\n```")

    return "\n\n".join(parts)


# ── Node implementations ───────────────────────────────────────────────────────
async def query_transform_node(state: RagState) -> RagState:
    """HyDE: generate a hypothetical code answer + extract any filename hints."""
    llm = get_llm()

    # Extract filenames from the original question BEFORE HyDE transform
    hints = extract_filename_hints(state["question"])
    if hints:
        logger.info("Filename hints detected in question: %s", hints)

    messages = _HYDE_PROMPT.format_messages(question=state["question"])
    response = await llm.ainvoke(messages)
    # Combine original + hypothetical for richer semantic search
    transformed = f"{state['question']}\n\n{response.content}"
    return {**state, "transformed_query": transformed, "filename_hints": hints}


async def retrieve_node(state: RagState) -> RagState:
    """
    Hybrid retrieval: semantic search + targeted filename search.

    When the question mentions a specific file name, chunks from that file are
    fetched directly via a Qdrant payload filter and merged with the semantic
    results. This ensures file-specific questions always get relevant context
    even when embedding similarity would favour README prose.
    """
    settings = get_settings()

    # ── Semantic search (always runs) ─────────────────────────────────────────
    semantic_docs = await similarity_search(
        query=state["transformed_query"],
        top_k=settings.retrieval_top_k,
    )

    # ── Filename-targeted search (runs only when hints are found) ─────────────
    file_docs: list[dict] = []
    for hint in state.get("filename_hints", []):
        results = await filename_search(hint, limit=30)
        file_docs.extend(results)

    # ── Merge + deduplicate by (source, start_line) ───────────────────────────
    seen: set[tuple] = set()
    merged: list[dict] = []

    # File-targeted results go first (higher priority in reranking pool)
    for doc in file_docs + semantic_docs:
        key = (doc["source"], doc["start_line"])
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    logger.info(
        "Retrieve: %d semantic + %d file-targeted = %d unique docs",
        len(semantic_docs), len(file_docs), len(merged),
    )
    return {**state, "raw_docs": merged}


async def rerank_node(state: RagState) -> RagState:
    """Cross-encoder reranks candidates — returns top-N most relevant."""
    settings = get_settings()
    # Reranking is CPU-bound; run in thread pool
    reranked = await asyncio.to_thread(
        rerank,
        state["question"],     # rerank against original question (not HyDE)
        state["raw_docs"],
        settings.rerank_top_n,
    )
    return {**state, "reranked_docs": reranked}


async def generate_node(state: RagState) -> RagState:
    """Synthesise a final answer from reranked code context."""
    llm = get_llm()
    context = _build_context(state["reranked_docs"])
    messages = _ANSWER_PROMPT.format_messages(
        context=context,
        question=state["question"],
    )
    content_chunks = []
    async for chunk in llm.astream(messages):
        content_chunks.append(chunk.content)
    answer = "".join(content_chunks)
    return {**state, "answer": answer}


# ── Graph assembly ────────────────────────────────────────────────────────────
def build_rag_graph() -> StateGraph:
    """Build and compile the LangGraph RAG state machine."""
    graph = StateGraph(RagState)

    graph.add_node("query_transform", query_transform_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("query_transform")
    graph.add_edge("query_transform", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# Compiled graph singleton
_rag_graph = None


def get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph


async def run_rag(question: str, chat_history: list[dict] | None = None) -> str:
    """
    Run the full RAG pipeline for a question and return the answer string.
    Non-streaming convenience wrapper.
    """
    graph = get_rag_graph()
    result = await graph.ainvoke({
        "question": question,
        "chat_history": chat_history or [],
        "transformed_query": "",
        "filename_hints": [],
        "raw_docs": [],
        "reranked_docs": [],
        "answer": "",
    })
    return result["answer"]


async def stream_rag(question: str, chat_history: list[dict] | None = None) -> AsyncIterator[str]:
    """
    Run the RAG pipeline and stream the final answer token-by-token via SSE.

    Yields status events for intermediate steps so the UI can show progress.
    """
    graph = get_rag_graph()
    inputs = {
        "question": question,
        "chat_history": chat_history or [],
        "transformed_query": "",
        "filename_hints": [],
        "raw_docs": [],
        "reranked_docs": [],
        "answer": "",
    }

    try:
        async for event in graph.astream_events(inputs, version="v2"):
            kind = event["event"]
            name = event["name"]

            # Capture node start events to yield status updates
            if kind == "on_chain_start" and name == "query_transform":
                yield "event: status\ndata: 🔍 Transforming query...\n\n"
            elif kind == "on_chain_start" and name == "retrieve":
                yield "event: status\ndata: 📂 Searching codebase...\n\n"
            elif kind == "on_chain_start" and name == "rerank":
                yield "event: status\ndata: 🎯 Reranking results...\n\n"
            elif kind == "on_chain_start" and name == "generate":
                yield "event: status\ndata: 💬 Generating answer...\n\n"

            # Capture chat model stream events to yield token chunks
            elif kind == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    # SSE spec: the data field must not contain raw newlines.
                    # Encode the token as JSON so newlines/special chars are safe.
                    encoded = json.dumps(token)
                    yield f"event: token\ndata: {encoded}\n\n"

        yield "event: done\ndata: [DONE]\n\n"
    except Exception as e:
        logger.exception("Error in stream_rag execution")
        raise e

