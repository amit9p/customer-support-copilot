"""
Multi-index SEARCH - query one index, or several together, with one global rerank.

    selected indexes --> each: dense + BM25 -> RRF (top N candidates per index)
                     --> merge all candidates
                     --> ONE cross-encoder rerank across everything
                     --> top 5

Why rerank once globally (not per index)? Scores from different indexes aren't comparable
(a #1 in ticket_index may be weaker than a #3 in policy_index). The cross-encoder gives every
candidate the same yardstick: relevance to the question.

In this file the index choice is made BY HAND. File 3 makes the copilot choose.

Run:  python v2_multi_index/2_multi_index_search.py
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_chroma import Chroma

from common import BM25Index, get_embeddings, load_all_docs, rerank, rrf, show_docs
from config import BM25_DIR, CHROMA_DIR, INDEXES, TOP_K

PER_INDEX_CANDIDATES = 8      # each index contributes up to this many to the global pool
DOCTYPE_TO_INDEX = {dt: name for name, dts in INDEXES.items() for dt in dts}


class MultiIndexSearcher:
    def __init__(self) -> None:
        emb = get_embeddings()
        self.stores: dict[str, Chroma] = {}
        self.bm25: dict[str, BM25Index] = {}
        for name in INDEXES:
            self.stores[name] = Chroma(collection_name=name, embedding_function=emb,
                                       persist_directory=str(CHROMA_DIR))
            pkl = BM25_DIR / f"{name}.pkl"
            if pkl.exists():
                self.bm25[name] = pickle.load(open(pkl, "rb"))
            else:                                   # fallback: rebuild from the collection
                self.bm25[name] = BM25Index(load_all_docs(self.stores[name]))

    def hybrid(self, question: str, index: str, k: int = PER_INDEX_CANDIDATES) -> list:
        dense = self.stores[index].similarity_search(question, k=k)
        sparse = self.bm25[index].search(question, k=k)
        pinned = self.bm25[index].pinned(question)
        fused = rrf([dense, sparse])
        return (pinned + [d for d in fused if d not in pinned])[:k]

    def search(self, question: str, indexes: list[str], k: int = TOP_K) -> list:
        candidates, seen = [], set()
        for name in indexes:
            for d in self.hybrid(question, name):
                if d.metadata["id"] not in seen:
                    seen.add(d.metadata["id"])
                    candidates.append(d)
        pinned = [d for d in candidates if d in sum((self.bm25[i].pinned(question) for i in indexes), [])]
        ranked = rerank(question, [d for d in candidates if d not in pinned], len(candidates))  # one global rerank

        # Coverage guarantee: when several indexes were routed, each index keeps its best doc in the
        # top k, then the remaining slots are filled by global rank. Without this, a 3-index question
        # (policy + order + ticket) can end up with zero order_guide docs in the top 5.
        final = list(pinned)
        if len(indexes) > 1:
            for name in indexes:
                best = next((d for d in ranked if DOCTYPE_TO_INDEX[d.metadata["doc_type"]] == name
                             and d not in final), None)
                if best is not None and len(final) < k:
                    final.append(best)
        final += [d for d in ranked if d not in final]
        return final[:k]


if __name__ == "__main__":
    q = ("A customer wants to change the address after the order shipped. "
         "What is the official process, and do we have a resolved case like this?")
    print(f"Question: {q}")
    searcher = MultiIndexSearcher()

    all4 = searcher.search(q, list(INDEXES))
    show_docs("Top 5 from ALL 4 indexes:", all4)

    two = searcher.search(q, ["order_index", "ticket_index"])
    show_docs("Top 5 from order_index + ticket_index only:", two)

    print("\nWhich is better: both lists should contain OG-002 and TCK-1004. The 2-index search is better "
          "in practice - same answer, fewer irrelevant policy/product candidates for the reranker to sort, "
          "and fewer embedding calls. That is exactly what the router in file 3 automates.")
