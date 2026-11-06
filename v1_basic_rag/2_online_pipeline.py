"""
V1 - ONLINE pipeline (runs every time a question is asked)

    question  -->  embed  -->  top 5 similar documents from Chroma  -->  LLM writes answer + sources

Like a librarian: find the 5 most relevant pages, hand them to the LLM, and the LLM
answers ONLY from those pages (or says "I don't know").

Run:  python v1_basic_rag/2_online_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config.py / common.py importable

from common import generate_answer, get_v1_store, print_result
from config import TOP_K


def retrieve(question: str, k: int = TOP_K):
    """Dense (embedding) similarity search - the simplest form of retrieval."""
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
