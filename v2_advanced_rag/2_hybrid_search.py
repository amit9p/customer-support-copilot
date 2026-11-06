"""
V2 technique 2 - HYBRID SEARCH (dense + BM25, merged with RRF)

Dense search understands MEANING ("refund timing" ~ "how long until money is back")
but is blind to exact tokens like "RR-004". BM25 is the opposite: dumb about meaning,
great at exact words. Run both and merge the two ranked lists with Reciprocal Rank Fusion.

    question --> dense top N  --\
                                 >-- RRF merge --> top 5
    question --> BM25  top N  --/

Run:  python v2_advanced_rag/2_hybrid_search.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import BM25Index, get_v1_store, load_all_docs, rrf, show_docs
from config import TOP_K

CANDIDATES = 10   # how many each searcher contributes before merging


class HybridSearcher:
    def __init__(self, store=None) -> None:
        self.store = store or get_v1_store()
        self.bm25 = BM25Index(load_all_docs(self.store))

    def search(self, question: str, k: int = TOP_K, filter: dict | None = None):
        dense = self.store.similarity_search(question, k=CANDIDATES, filter=filter)
        sparse = self.bm25.search(question, k=CANDIDATES)
        fused = rrf([dense, sparse])
        # RRF rewards docs found by BOTH searchers, so a doc only BM25 found (even at rank 1) can
        # fall out of the top 5. An exact ID match is a certainty, so it is pinned ahead of the fusion.
        pinned = self.bm25.pinned(question)
        return (pinned + [d for d in fused if d not in pinned])[:k]


if __name__ == "__main__":
    q = "What does policy RR-004 say?"
    print(f"Question: {q}")

    store = get_v1_store()
    show_docs("BEFORE (V1 dense only):", store.similarity_search(q, k=TOP_K))

    hybrid = HybridSearcher(store)
    show_docs("BM25 only (for comparison):", hybrid.bm25.search(q, k=TOP_K))
    show_docs("AFTER (hybrid = dense + BM25 -> RRF):", hybrid.search(q))

    print("\nWhat changed: embeddings don't 'know' that RR-004 is an ID, so dense search returned other "
          "return/refund policies. BM25 matched the exact token 'rr-004'; because dense never found it, "
          "plain RRF would still drop it, so exact-ID matches are pinned ahead of the fused list.")
