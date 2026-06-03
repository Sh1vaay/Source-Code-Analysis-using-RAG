"""
src/evaluation/eval.py
───────────────────────
RAGAS evaluation suite for the RAG pipeline.

Measures three key metrics on a small golden dataset:
  • faithfulness    — Does the answer contradict the retrieved context?
  • answer_relevancy — Is the answer on-topic with the question?
  • context_recall  — Was the ground-truth info present in retrieved docs?

Usage:
    python -m src.evaluation.eval

Set RAGAS_OPENAI_API_KEY if you want to use GPT-4 as the RAGAS judge.
Otherwise it falls back to Groq/Ollama (whatever is configured in .env).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.chains.graph import run_rag
from src.retrieval.vector_store import similarity_search

# ── Golden dataset ─────────────────────────────────────────────────────────────
# Edit this file with real Q&A pairs after indexing a repository.
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

SAMPLE_GOLDEN_DATASET = [
    {
        "question": "What does the DataIngestion class do?",
        "ground_truth": "DataIngestion downloads and extracts zip files based on configuration.",
    },
    {
        "question": "How is the model trained?",
        "ground_truth": "The model is trained using ElasticNet from sklearn with alpha and l1_ratio parameters.",
    },
]


async def _collect_rag_outputs(dataset: list[dict]) -> list[dict]:
    """Run RAG for each question and collect outputs."""
    results = []
    for item in dataset:
        question = item["question"]
        print(f"  Evaluating: {question[:60]}...")

        # Get answer
        answer = await run_rag(question)

        # Get retrieved contexts
        docs = await similarity_search(question, top_k=5)
        contexts = [d["content"] for d in docs]

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item.get("ground_truth", ""),
        })
    return results


def run_ragas_evaluation(results: list[dict]) -> dict:
    """Run RAGAS metrics on collected results."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness

        from src.chains.llm_factory import get_llm
        from src.retrieval.embeddings import get_embedder
        from langchain_core.embeddings import Embeddings

        class RagasEmbeddings(Embeddings):
            def __init__(self, e):
                self.e = e
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self.e.embed(texts)
            def embed_query(self, text: str) -> list[float]:
                return self.e.embed_query(text)

        llm = get_llm()
        embeddings = RagasEmbeddings(get_embedder())

        dataset = Dataset.from_list(results)
        score = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
        return score.to_pandas().mean().to_dict()

    except ImportError:
        print("⚠️  ragas or datasets not installed. Run: pip install ragas datasets")
        return {}
    except Exception as e:
        print(f"⚠️  RAGAS evaluation failed: {e}")
        return {}


async def main():
    print("🧪  Starting RAGAS Evaluation\n")

    # Load golden dataset
    if GOLDEN_DATASET_PATH.exists():
        with open(GOLDEN_DATASET_PATH) as f:
            dataset = json.load(f)
        print(f"📂  Loaded {len(dataset)} items from {GOLDEN_DATASET_PATH}")
    else:
        dataset = SAMPLE_GOLDEN_DATASET
        print(f"📂  Using {len(dataset)} sample questions (no golden_dataset.json found)")

    print("\n⏳  Running RAG pipeline on each question...")
    results = await _collect_rag_outputs(dataset)

    print("\n📊  Computing RAGAS metrics...")
    scores = run_ragas_evaluation(results)

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    if scores:
        for metric, value in scores.items():
            print(f"  {metric:<30}: {value:.3f}")
    else:
        print("  (Install ragas to get full metrics)")

    print("\n  Raw answers:")
    for r in results:
        print(f"\n  Q: {r['question']}")
        print(f"  A: {r['answer'][:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
