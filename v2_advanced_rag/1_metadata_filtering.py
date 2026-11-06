"""
V2 technique 1: metadata filtering.
The LLM extracts doc_type / product / category from the question and Chroma searches only
documents matching that filter. Prints before vs after for the demo question.
Run: python v2_advanced_rag/1_metadata_filtering.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage

from common import get_llm, get_v1_store, show_docs
from config import TOP_K

# The values the LLM may choose from - taken straight from the dataset.
DOC_TYPES = ["product_faq", "return_policy", "shipping_policy", "order_guide", "known_issue", "resolved_ticket"]
PRODUCTS = ["iPhone 16", "Galaxy S25", "OnePlus 13", "Pixel 9", "MacBook Air M3", "XPS 13", "Pavilion 15",
            "ThinkPad E14", "AirPods Pro 2", "Galaxy Buds3 Pro", "WH-1000XM5", "Anker 20W Charger", "MX Master 3S"]
CATEGORIES = ["box_contents", "warranty", "payments", "exchange", "specifications", "software", "setup",
              "compatibility", "returns", "refunds", "delivery", "order_changes", "tracking",
              "connectivity", "charging"]

FILTER_PROMPT = f"""Extract search filters from a customer-support question. Return JSON only, with keys
doc_type, product, category. Use null when the question does not clearly imply a value.
Allowed doc_type: {DOC_TYPES}
Allowed product: {PRODUCTS}
Allowed category: {CATEGORIES}
Hints: "known issue"/"problem" -> known_issue; "policy" -> return_policy or shipping_policy;
"has this happened before"/"resolved case" -> resolved_ticket; "how do I" -> order_guide."""


def extract_filter(question: str) -> dict:
    """Ask the LLM which metadata values the question implies. Returns only non-null keys."""
    msg = get_llm().invoke([SystemMessage(content=FILTER_PROMPT), HumanMessage(content=question)]).content
    try:
        raw = json.loads(msg.strip().strip("`").removeprefix("json"))
    except json.JSONDecodeError:
        return {}
    return {k: v for k, v in raw.items() if v}


def to_chroma_filter(flt: dict) -> dict | None:
    """Chroma wants {"key": value} for one condition and {"$and": [...]} for several."""
    conds = [{k: v} for k, v in flt.items()]
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def filtered_search(question: str, k: int = TOP_K):
    flt = extract_filter(question)
    return get_v1_store().similarity_search(question, k=k, filter=to_chroma_filter(flt)), flt


if __name__ == "__main__":
    q = "Any known issues with the Galaxy Buds3 Pro?"
    print(f"Question: {q}")

    before = get_v1_store().similarity_search(q, k=TOP_K)
    show_docs("BEFORE (V1 dense search, no filter):", before)

    after, flt = filtered_search(q)
    show_docs(f"AFTER (metadata filter {flt}):", after)

    print("\nWhat changed: the filter removed FAQ/ticket docs about the Buds that are not 'known issues', "
          "so only the actual known-issue record(s) for this product are returned - fewer distractions for the LLM.")
