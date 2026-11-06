"""
Intent router.
The LLM classifies a question into intents (policy, product, order, past_issue, out_of_scope)
and Python maps intents to indexes. Returns JSON {intents, indexes, reason}.
If the LLM fails or returns bad JSON, all 4 indexes are searched. Never raises.
Run: python v2_multi_index/3_intent_router.py   (tests the router on the 7 test questions)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage

from common import get_llm
from config import INDEXES

INTENT_TO_INDEX = {
    "policy": "policy_index",
    "product": "product_index",
    "order": "order_index",
    "past_issue": "ticket_index",
}

ROUTER_PROMPT = """You route questions for Dandes Electronics, an online store selling phones, laptops and
accessories (iPhone 16, Galaxy S25, OnePlus 13, Pixel 9, MacBook Air M3, Dell XPS 13, HP Pavilion 15,
ThinkPad E14, AirPods Pro 2, Galaxy Buds3 Pro, Sony WH-1000XM5, Anker 20W Charger, Logitech MX Master 3S).

Classify the support question into one or more intents:
- policy       : rules about returns, refunds, replacements, delivery times/charges, express delivery,
                 Cash on Delivery limits, failed deliveries, or a question naming a policy code (RR-xxx, SH-xxx)
- product      : product features, box contents, warranty, compatibility, EMI, exchange, known problems/fixes
- order        : how to do something with an order: cancel, change address, track, failed payment,
                 report a damaged/wrong item, raise a warranty claim
- past_issue   : "has this happened before", "resolved case", "previous customer", past tickets
- out_of_scope : not about this store's products, orders or policies (other products, discounts/coupons,
                 repairs the store doesn't do, general knowledge). The knowledge base only covers these
                 payment methods: UPI, wallets, credit/debit cards, net banking, EMI and Cash on Delivery -
                 any other payment method is out_of_scope.

A question can have several intents. Use out_of_scope ALONE, never with other intents.
Return JSON only: {"intents": [...], "reason": "<one short sentence>"}"""

ALL_INDEXES = list(INDEXES)


def route(question: str) -> dict:
    """Always returns {"intents", "indexes", "reason"}; never raises."""
    try:
        text = get_llm().invoke([SystemMessage(content=ROUTER_PROMPT),
                                 HumanMessage(content=question)]).content.strip()
        text = text.strip("`").removeprefix("json").strip()
        data = json.loads(text)
        intents = [i for i in data.get("intents", []) if i in INTENT_TO_INDEX or i == "out_of_scope"]
        if not intents:
            raise ValueError("no valid intents")
        if "out_of_scope" in intents:
            return {"intents": ["out_of_scope"], "indexes": [], "reason": data.get("reason", "")}
        indexes = [INTENT_TO_INDEX[i] for i in intents]
        return {"intents": intents, "indexes": indexes, "reason": data.get("reason", "")}
    except Exception as e:                       # bad JSON, API error, anything -> safe fallback
        return {"intents": ["fallback"], "indexes": ALL_INDEXES,
                "reason": f"router failed ({type(e).__name__}); searching all indexes"}


if __name__ == "__main__":
    TESTS = [
        ("Can I return an opened laptop on day 8 if I have the original box and all accessories?", ["policy_index"]),
        ("My Galaxy Buds3 Pro left earbud won't connect. Is this a known issue, and has the same problem been resolved for another customer? What fixed it?", ["product_index", "ticket_index"]),
        ("How long does delivery take to a remote pincode, and is it free?", ["policy_index"]),
        ("A customer wants to change the address after the order shipped. What is the official process, and do we have a resolved case like this?", ["order_index", "ticket_index"]),
        ("What does policy RR-004 say?", ["policy_index"]),
        ("A customer received a damaged phone and wants a refund instead of a replacement. What is the official process, and do we have a previous resolved case where a refund was issued?", ["policy_index", "order_index", "ticket_index"]),
        ("Do you sell refrigerators?", []),
    ]
    hits = 0
    for i, (q, expected) in enumerate(TESTS, 1):
        r = route(q)
        ok = set(expected) <= set(r["indexes"]) and len(set(r["indexes"]) - set(expected)) <= 1
        if not expected:
            ok = r["indexes"] == []
        hits += ok
        print(f"\nQ{i}: {q[:80]}{'...' if len(q) > 80 else ''}")
        print(f"   predicted : {r['indexes'] or 'none'}   intents={r['intents']}")
        print(f"   expected  : {expected or 'none'}   {'OK' if ok else 'MISS'}")
        print(f"   reason    : {r['reason']}")
    print(f"\nRouter accuracy on test questions: {hits}/{len(TESTS)}")
