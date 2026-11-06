"""
Helpers shared by all pipeline files: Chroma access, prompt building, answer generation,
BM25 index, Reciprocal Rank Fusion, cross-encoder reranking.
"""
from __future__ import annotations

import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import CHROMA_DIR, EMBEDDING_MODEL, IDK, LLM_MODEL, V1_COLLECTION


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm() -> ChatOpenAI:
    # gpt-6-luna only accepts its default temperature, so we don't set one
    return ChatOpenAI(model=LLM_MODEL)


def get_v1_store() -> Chroma:
    return Chroma(
        collection_name=V1_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def format_context(docs: list[Document]) -> str:
    """Format retrieved docs as text blocks that start with the doc ID."""
    blocks = []
    for d in docs:
        m = d.metadata
        blocks.append(f"[{m['id']}] (type={m['doc_type']}, date={m.get('date', 'n/a')})\n{d.page_content}")
    return "\n\n".join(blocks)


SYSTEM_PROMPT = f"""You are a customer support copilot for Dandes Electronics.
Answer the support executive's question using ONLY the documents provided.

Rules:
- Be concise and specific (numbers, days, steps).
- End with one line in the form:  Sources: ID1, ID2   (only IDs you actually used)
- If the documents do not contain the answer, reply with exactly this sentence and nothing else:
{IDK}
"""


LATEST_WINS_RULE = """
- Documents can disagree (e.g. an old ticket says 7 days, the current policy says 10 days).
  ONLY when they disagree: follow the document with the LATEST date and mention that date, e.g.
  "Under the current policy effective 1 July 2026, ...". Policies beat tickets when dates are equal.
  When the documents agree, do not mention dates at all.
"""


def load_module(rel_path: str):
    """Import a pipeline file by path (names start with digits, so a normal import won't work)."""
    import importlib.util
    from config import PROJECT_ROOT
    path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_answer(question: str, docs: list[Document], latest_wins: bool = False) -> tuple[str, list[str]]:
    """Return (answer, source_ids). The answer never includes the 'Sources:' line itself."""
    system = SYSTEM_PROMPT + (LATEST_WINS_RULE if latest_wins else "")
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Documents:\n{format_context(docs)}\n\nQuestion: {question}"),
    ]
    text = get_llm().invoke(messages).content.strip()

    if IDK in text:
        return IDK, []

    sources: list[str] = []
    match = re.search(r"Sources?\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        sources = [s.strip() for s in match.group(1).split(",") if s.strip()]
        text = text[: match.start()].rstrip()
    return text, sources


def print_result(question: str, answer: str, sources: list[str]) -> None:
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"Sources: {', '.join(sources) if sources else 'none'}")


def show_docs(label: str, docs: list[Document]) -> None:
    """Print a ranked list of retrieved docs - used for the before/after demos in V2."""
    print(f"\n{label}")
    for i, d in enumerate(docs, 1):
        m = d.metadata
        title = d.page_content.split("\n", 1)[0]
        print(f"  {i}. {m['id']:<9} {m['doc_type']:<15} {m.get('date', ''):<10} {title}")


# ---------------------------------------------------------------------------
# Sparse search (BM25) + Reciprocal Rank Fusion - shared by V2 and multi-index
# ---------------------------------------------------------------------------
import re as _re

from rank_bm25 import BM25Okapi


def load_all_docs(store: Chroma) -> list[Document]:
    """Pull every document (text + metadata) back out of a Chroma collection."""
    data = store.get(include=["documents", "metadatas"])
    return [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]


STOPWORDS = {"what", "does", "do", "is", "are", "the", "a", "an", "to", "of", "for", "and", "or", "in",
             "on", "it", "i", "my", "can", "say", "says", "with", "this", "that", "be", "how", "we", "have",
             "has", "any", "if", "after", "before", "about", "there", "please", "customer"}


def _tokenize(text: str) -> list[str]:
    # lower-case words minus stopwords; keeps "rr-004" as one token so exact codes match
    return [w for w in _re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()) if w not in STOPWORDS]


class BM25Index:
    """BM25 keyword search over the documents.
    Exact-ID rule: if a query token equals a document ID (e.g. rr-004) that doc is pinned to the
    top - plain BM25 ranked OG-001 above RR-004 for "What does policy RR-004 say?" (see report Q2).
    """

    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.by_id = {d.metadata["id"].lower(): d for d in docs}
        self.bm25 = BM25Okapi([_tokenize(d.metadata["id"] + " " + d.page_content) for d in docs])

    def pinned(self, query: str) -> list[Document]:
        """Docs whose ID appears verbatim in the query (e.g. 'RR-004')."""
        return [self.by_id[t] for t in _tokenize(query) if t in self.by_id]

    def search(self, query: str, k: int) -> list[Document]:
        tokens = _tokenize(query)
        pinned = self.pinned(query)                                          # exact ID matches first
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        rest = [self.docs[i] for i in ranked if scores[i] > 0 and self.docs[i] not in pinned]
        return (pinned + rest)[:k]


def rrf(result_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion: score(doc) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for docs in result_lists:
        for rank, d in enumerate(docs, 1):
            did = d.metadata["id"]
            by_id[did] = d
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
    return [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)]


# ---------------------------------------------------------------------------
# Cross-encoder reranking - shared by V2 and multi-index
# ---------------------------------------------------------------------------
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Load the cross-encoder once."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(question: str, docs: list[Document], k: int) -> list[Document]:
    """Score each (question, doc) pair with the cross-encoder and return the best k."""
    if not docs:
        return []
    scores = get_reranker().predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:k]]
