"""
V2 technique 3: reranking.
Retrieve the top 20 by embedding, score each (question, document) pair with a cross-encoder,
keep the best 5. Prints before vs after for the demo question.
Run: python v2_advanced_rag/3_reranking.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_v1_store, rerank, show_docs
from config import RERANK_CANDIDATES, TOP_K


def retrieve_and_rerank(question: str, store=None, k: int = TOP_K):
    store = store or get_v1_store()
    candidates = store.similarity_search(question, k=RERANK_CANDIDATES)   # top 20
    return rerank(question, candidates, k)                                 # best 5


if __name__ == "__main__":
    q = "Customer got a damaged phone and wants a refund. What's the process?"
    print(f"Question: {q}")

    store = get_v1_store()
    before = store.similarity_search(q, k=TOP_K)
    after = retrieve_and_rerank(q, store)
    show_docs("BEFORE (V1 dense top 5):", before)
    show_docs("AFTER (dense top 20 -> cross-encoder -> top 5):", after)

    ids_before = {d.metadata["id"] for d in before}
    ids_after = {d.metadata["id"] for d in after}
    print(f"\nWhat changed: dropped {sorted(ids_before - ids_after) or 'nothing'}, "
          f"added {sorted(ids_after - ids_before) or 'nothing'}. The cross-encoder reads question and doc "
          "together, so loosely related tickets lose to docs that answer the question; on a 50-doc "
          "collection the shift is small - it matters more when merging candidates from several indexes.")
