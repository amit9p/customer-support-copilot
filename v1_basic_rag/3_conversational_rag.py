"""
V1 - CONVERSATIONAL RAG (follow-up questions work)

Problem: "What if the box is opened?" means nothing on its own - the retriever would
search for "box opened" and miss that we were talking about laptops.

Fix (two steps):
  1. Rewrite the follow-up into a standalone question using the chat history
        "What if the box is opened?"  -->  "Can an opened laptop box be returned?"
  2. Run the normal online pipeline on the rewritten question.

Run:  python v1_basic_rag/3_conversational_rag.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config.py / common.py importable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from common import generate_answer, get_llm, get_v1_store, print_result
from config import TOP_K

REWRITE_PROMPT = """Given the chat history and a follow-up question, rewrite the follow-up as a
standalone question that makes sense without the history. Keep it short.
If it is already standalone, return it unchanged. Return only the question."""


def make_standalone(question: str, history: list) -> str:
    if not history:
        return question
    messages = [SystemMessage(content=REWRITE_PROMPT), *history,
                HumanMessage(content=f"Follow-up question: {question}")]
    return get_llm().invoke(messages).content.strip()


class Chat:
    """Keeps the history. Call .ask() repeatedly like a real conversation."""

    def __init__(self) -> None:
        self.history: list = []
        self.store = get_v1_store()

    def ask(self, question: str) -> tuple[str, list[str]]:
        standalone = make_standalone(question, self.history)
        self.last_rewrite = standalone if standalone != question else None

        docs = self.store.similarity_search(standalone, k=TOP_K)
        answer, sources = generate_answer(standalone, docs)

        self.history += [HumanMessage(content=question), AIMessage(content=answer)]
        return answer, sources


if __name__ == "__main__":
    chat = Chat()
    for q in [
        "What is the return window for laptops?",
        "What if the box is opened?",          # "the box" must be understood as the laptop box
        "And how many days for a refund on UPI?",
    ]:
        answer, sources = chat.ask(q)
        print_result(q, answer, sources)
        if chat.last_rewrite:
            print(f"  (searched as: {chat.last_rewrite})")
