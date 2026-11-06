"""
V2 multi-index + routing pipeline (final version).
route -> hybrid search in each chosen index -> merge -> one global rerank -> top 5 -> answer.
out_of_scope skips retrieval and returns the fixed "I don't know" sentence.
The routing decision is printed before every answer.
Run: python v2_multi_index/4_v2_multi_index_pipeline.py          (7 test questions)
     python v2_multi_index/4_v2_multi_index_pipeline.py --save   (also writes demo_output.txt)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import generate_answer, load_module
from config import IDK, PROJECT_ROOT

router = load_module("v2_multi_index/3_intent_router.py")
mis = load_module("v2_multi_index/2_multi_index_search.py")

_searcher = None


def run(question: str) -> dict:
    """Standard contract for the evaluation scripts."""
    global _searcher
    routing = router.route(question)

    if routing["intents"] == ["out_of_scope"]:
        return {"answer": IDK, "sources": [], "retrieved_ids": [], "routing": routing}

    if _searcher is None:
        _searcher = mis.MultiIndexSearcher()
    docs = _searcher.search(question, routing["indexes"])
    answer, sources = generate_answer(question, docs, latest_wins=True)
    return {"answer": answer, "sources": sources,
            "retrieved_ids": [d.metadata["id"] for d in docs], "routing": routing}


def format_run(question: str) -> str:
    out = run(question)
    r = out["routing"]
    return (f"\nQuestion : {question}\n"
            f"Routing  : intents={r['intents']} -> indexes={r['indexes'] or 'none'}\n"
            f"Answer   : {out['answer']}\n"
            f"Sources  : {', '.join(out['sources']) if out['sources'] else 'none'}\n")


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
    blocks = []
    for q in TEST_QUESTIONS:
        block = format_run(q)
        print(block)
        blocks.append(block)
    if "--save" in sys.argv:
        out_file = PROJECT_ROOT / "demo_output.txt"
        out_file.write_text("".join(blocks).lstrip("\n"), encoding="utf-8")
        print(f"Saved {out_file}")
