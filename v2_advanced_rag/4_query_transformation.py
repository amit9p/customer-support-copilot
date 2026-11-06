"""
V2 technique 4 - QUERY TRANSFORMATION

Customers (and support execs) ask vague questions. "My order hasn't come yet" could be about
delivery times, tracking, failed delivery attempts or remote pincodes. Two fixes:

  (A) Query rewriting   : LLM turns the vague question into one precise search query
  (B) Multi-query       : LLM writes 3 different versions of the question, we search with each,
                          and merge the result lists with RRF - covers more angles

Run:  python v2_advanced_rag/4_query_transformation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage

from common import get_llm, get_v1_store, rrf, show_docs
from config import TOP_K


def diff_line(before, after) -> str:
    b = {d.metadata["id"] for d in before}
    a = {d.metadata["id"] for d in after}
    return f"dropped {sorted(b - a) or 'nothing'}, added {sorted(a - b) or 'nothing'}"

REWRITE_PROMPT = """You improve search queries for an electronics store's support knowledge base
(topics: delivery times, tracking, returns, refunds, order changes, product issues, past tickets).
Rewrite the user's question as ONE clear, specific search query using the store's vocabulary.
Return only the query."""

MULTI_PROMPT = """You improve search queries for an electronics store's support knowledge base
(topics: delivery times, tracking, returns, refunds, order changes, product issues, past tickets).
Write 3 different versions of the user's question, each focusing on a different possible meaning
or using different wording. One per line, no numbering, nothing else."""


def rewrite_query(question: str) -> str:
    return get_llm().invoke([SystemMessage(content=REWRITE_PROMPT), HumanMessage(content=question)]).content.strip()


def multi_queries(question: str) -> list[str]:
    text = get_llm().invoke([SystemMessage(content=MULTI_PROMPT), HumanMessage(content=question)]).content
    variants = [line.strip("-* ").strip() for line in text.splitlines() if line.strip()]
    return [question] + variants[:3]            # always keep the original too


def multi_query_search(question: str, store=None, k: int = TOP_K):
    store = store or get_v1_store()
    queries = multi_queries(question)
    result_lists = [store.similarity_search(q, k=k) for q in queries]
    return rrf(result_lists)[:k], queries


if __name__ == "__main__":
    q = "My order hasn't come yet"
    store = get_v1_store()
    print(f"Question: {q}")
    before = store.similarity_search(q, k=TOP_K)
    show_docs("BEFORE (V1 dense search on the raw question):", before)

    # ---- Demo A: query rewriting ------------------------------------------------
    rewritten = rewrite_query(q)
    print(f"\n(A) Rewritten query: {rewritten}")
    after_a = store.similarity_search(rewritten, k=TOP_K)
    show_docs("AFTER (A) - search on the rewritten query:", after_a)
    print(f"What changed: {diff_line(before, after_a)}. Rewriting replaces the question with the LLM's "
          "interpretation - good when the original is vague, risky when the LLM guesses the wrong meaning.")

    # ---- Demo B: multi-query retrieval ------------------------------------------
    docs, queries = multi_query_search(q, store)
    print("\n(B) Queries used:")
    for s in queries:
        print(f"   - {s}")
    show_docs("AFTER (B) - 4 searches merged with RRF:", docs)
    print(f"What changed: {diff_line(before, docs)}. Multi-query keeps the original question AND adds "
          "variants, then RRF keeps what several of them agree on - it can only add angles, never drift. "
          "That is why the V2 pipeline uses (B).")
