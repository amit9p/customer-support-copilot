"""
V2 pipeline - all four techniques combined on the single 'dandes_v1' collection.
This is the version evaluated as "V2 (single index)".

Order of steps (and why):
  1. Multi-query        - widen the net first: 4 versions of the question
  2. Hybrid search      - dense + BM25 for EACH version, all lists merged with one RRF
  3. Metadata filter    - as a SOFT boost: a filtered dense search is one more list in the RRF.
                          A hard filter would break multi-part questions (policy + guide + ticket).
  4. Rerank             - cross-encoder scores the top 20 candidates, keep 5 (exact-ID hits stay pinned)
  5. Answer             - "latest date wins" rule added to the prompt

Run:  python v2_advanced_rag/5_v2_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (BM25Index, generate_answer, load_all_docs, load_module, get_v1_store, print_result,
                    rerank, rrf, show_docs)
from config import RERANK_CANDIDATES, TOP_K

meta_filter = load_module("v2_advanced_rag/1_metadata_filtering.py")      # extract_filter, to_chroma_filter
query_tx = load_module("v2_advanced_rag/4_query_transformation.py")       # multi_queries

CANDIDATES_PER_SEARCH = 10


class V2Pipeline:
    def __init__(self) -> None:
        self.store = get_v1_store()
        self.bm25 = BM25Index(load_all_docs(self.store))

    def retrieve(self, question: str, k: int = TOP_K) -> list:
        # 1. multi-query
        queries = query_tx.multi_queries(question)

        # 2. hybrid search per query -> one big RRF
        lists = []
        for q in queries:
            lists.append(self.store.similarity_search(q, k=CANDIDATES_PER_SEARCH))
            lists.append(self.bm25.search(q, k=CANDIDATES_PER_SEARCH))

        # 3. metadata filter as a soft boost (only if the LLM found a filter)
        flt = meta_filter.extract_filter(question)
        if flt:
            lists.append(self.store.similarity_search(question, k=CANDIDATES_PER_SEARCH,
                                                      filter=meta_filter.to_chroma_filter(flt)))

        candidates = rrf(lists)[:RERANK_CANDIDATES]

        # 4. rerank, keeping exact-ID matches on top
        pinned = self.bm25.pinned(question)
        reranked = rerank(question, [d for d in candidates if d not in pinned], k)
        return (pinned + reranked)[:k]

    def run(self, question: str) -> dict:
        docs = self.retrieve(question)
        answer, sources = generate_answer(question, docs, latest_wins=True)   # 5. answer
        return {"answer": answer, "sources": sources,
                "retrieved_ids": [d.metadata["id"] for d in docs], "routing": None}


_pipeline: V2Pipeline | None = None


def run(question: str) -> dict:
    """Standard contract for the evaluation scripts."""
    global _pipeline
    if _pipeline is None:
        _pipeline = V2Pipeline()
    return _pipeline.run(question)


if __name__ == "__main__":
    TEST_QUESTIONS = [
        "Can I return an opened laptop on day 8 if I have the original box and all accessories?",
        "My Galaxy Buds3 Pro left earbud won't connect. Is this a known issue, and has the same problem been resolved for another customer? What fixed it?",
        "How long does delivery take to a remote pincode, and is it free?",
        "A customer wants to change the address after the order shipped. What is the official process, and do we have a resolved case like this?",
        "What does policy RR-004 say?",
        "A customer received a damaged phone and wants a refund instead of a replacement. What is the official process, and do we have a previous resolved case where a refund was issued?",
        "Do you sell refrigerators?",
    ]
    for q in TEST_QUESTIONS:
        out = run(q)
        print_result(q, out["answer"], out["sources"])
        print(f"  (retrieved: {', '.join(out['retrieved_ids'])})")
