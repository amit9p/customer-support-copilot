"""
Builds evaluation/comparison_table.csv from the retrieval and generation metrics.
Run: python evaluation/4_comparison_table.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EVAL_DIR, RESULTS_DIR  # noqa: E402

VERSIONS = ["V1 (basic)", "V2 (single index)", "V2 (multi-index + routing)"]
ROWS = [
    "Routing Accuracy", "Precision@5", "Recall@5", "Coverage (multi-source)",
    "Faithfulness (own judge)", "Faithfulness (RAGAS)", "Answer Relevancy",
    "Context Precision", "Context Recall", "Refusal Accuracy", "False Refusals",
]


def main() -> None:
    retrieval = json.loads((RESULTS_DIR / "retrieval_metrics.json").read_text(encoding="utf-8"))
    gen_path = RESULTS_DIR / "generation_metrics.json"
    generation = json.loads(gen_path.read_text(encoding="utf-8")) if gen_path.exists() else {}

    table = {}
    for v in VERSIONS:
        merged = {**retrieval.get(v, {}), **generation.get(v, {})}
        table[v] = [merged.get(row, "n/a") for row in ROWS]

    df = pd.DataFrame(table, index=ROWS)
    df.index.name = "Metric"
    df.to_csv(EVAL_DIR / "comparison_table.csv")
    print(df.to_string())
    print(f"\nSaved {EVAL_DIR / 'comparison_table.csv'}")


if __name__ == "__main__":
    main()
