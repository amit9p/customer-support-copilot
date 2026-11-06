"""
V1 online pipeline.
Takes a question, retrieves the top 5 documents from 'dandes_v1' by embedding similarity,
and asks the LLM to answer from those documents only, with source IDs.
Run: python v1_basic_rag/2_online_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config.py / common.py importable

from common import generate_answer, get_v1_store, print_result
from config import TOP_K


def retrieve(question: str, k: int = TOP_K):
    """Embedding similarity search."""
    return get_v1_store().similarity_search(question, k=k)


def ask(question: str) -> tuple[str, list[str]]:
    docs = retrieve(question)
    return generate_answer(question, docs)


def run(question: str) -> dict:
    """Standard contract used by the evaluation scripts (same shape for every version)."""
    docs = retrieve(question)
    answer, sources = generate_answer(question, docs)
    return {
        "answer": answer,
        "sources": sources,
        "retrieved_ids": [d.metadata["id"] for d in docs],
        "routing": None,                     # V1 has no router
    }


if __name__ == "__main__":
    demo_questions = [
        "How long do UPI refunds take?",
        "What is the return window for laptops?",
        "Do you sell refrigerators?",           # should trigger the exact "I don't know" sentence
    ]
    for q in demo_questions:
        answer, sources = ask(q)
        print_result(q, answer, sources)
