"""
Retrieval metrics - did we fetch the RIGHT documents, and did the router/refusal behave?

    Precision@5      relevant docs in top 5 / 5                  (in-scope questions)
    Recall@5         relevant docs in top 5 / expected sources   (in-scope questions)
    Coverage         same as recall, but only on multi-source questions (2+ indexes)
    Routing Accuracy router searched all expected indexes and at most one extra;
                     out-of-scope is correct only if the router said out_of_scope
    Refusal Accuracy % of out-of-scope questions answered with the exact IDK sentence
    False Refusals   number of in-scope questions wrongly answered with IDK

Run:  python evaluation/2_retrieval_metrics.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EVAL_DIR, IDK, RESULTS_DIR, TOP_K  # noqa: E402

RESULT_FILES = {
    "V1 (basic)": "v1_results.json",
    "V2 (single index)": "v2_single_index_results.json",
    "V2 (multi-index + routing)": "v2_multi_index_results.json",
}


def routing_correct(expected: list[str], routing: dict | None) -> bool:
    if routing is None:
        return False
    searched = set(routing.get("indexes", []))
    if not expected:                                   # out-of-scope question
        return "out_of_scope" in routing.get("intents", []) and not searched
    exp = set(expected)
    return exp <= searched and len(searched - exp) <= 1


def compute(results: list[dict], golden: dict[str, dict]) -> dict:
    p_at_k, r_at_k, coverage, routing_hits, refusal_hits = [], [], [], [], []
    false_refusals = 0
    has_router = any(r.get("routing") for r in results)

    for r in results:
        g = golden[r["id"]]
        expected = set(g["expected_sources"])
        in_scope = bool(expected)
        refused = r["answer"].strip() == IDK

        if has_router:
            routing_hits.append(routing_correct(g["expected_indexes"], r.get("routing")))

        if not in_scope:
            refusal_hits.append(refused)
            continue

        if refused:
            false_refusals += 1

        retrieved = set(r["retrieved_ids"][:TOP_K])        # unique IDs in the top 5
        hits = len(retrieved & expected)
        p_at_k.append(hits / TOP_K)
        r_at_k.append(hits / len(expected))
        if len(g["expected_indexes"]) >= 2:
            coverage.append(hits / len(expected))

    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else None   # noqa: E731
    return {
        "Routing Accuracy": avg(routing_hits) if has_router else "n/a",
        "Precision@5": avg(p_at_k),
        "Recall@5": avg(r_at_k),
        "Coverage (multi-source)": avg(coverage),
        "Refusal Accuracy": avg(refusal_hits),
        "False Refusals": false_refusals,
    }


def main() -> None:
    golden = {q["id"]: q for q in json.loads((EVAL_DIR / "golden_test_set.json").read_text(encoding="utf-8"))["questions"]}
    all_scores = {}
    for label, fname in RESULT_FILES.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"{label}: no results file yet ({fname})")
            continue
        scores = compute(json.loads(path.read_text(encoding="utf-8")), golden)
        all_scores[label] = scores
        print(f"\n{label}")
        for k, v in scores.items():
            print(f"  {k:<26} {v}")

    (RESULTS_DIR / "retrieval_metrics.json").write_text(json.dumps(all_scores, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS_DIR / 'retrieval_metrics.json'}")


if __name__ == "__main__":
    main()
