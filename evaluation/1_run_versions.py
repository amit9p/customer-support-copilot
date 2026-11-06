"""
Run the 20 golden questions through every pipeline version and save the raw outputs.

    evaluation/results/v1_results.json
    evaluation/results/v2_single_index_results.json
    evaluation/results/v2_multi_index_results.json

Each pipeline file exposes  run(question) -> {"answer", "sources", "retrieved_ids", "routing"}.
Versions whose file is not written yet are skipped, so this can be run early for the V1 baseline.

Run:  python evaluation/1_run_versions.py            (all available versions)
      python evaluation/1_run_versions.py v1          (one version)
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EVAL_DIR, RESULTS_DIR  # noqa: E402

VERSIONS = {
    "v1": ("v1_basic_rag/2_online_pipeline.py", "v1_results.json"),
    "v2_single": ("v2_advanced_rag/5_v2_pipeline.py", "v2_single_index_results.json"),
    "v2_multi": ("v2_multi_index/4_v2_multi_index_pipeline.py", "v2_multi_index_results.json"),
}


def load_pipeline(rel_path: str):
    """Import a pipeline file by path (file names start with a digit, so a normal import won't work)."""
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(selected: list[str]) -> None:
    questions = json.loads((EVAL_DIR / "golden_test_set.json").read_text(encoding="utf-8"))["questions"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name in selected:
        rel_path, out_file = VERSIONS[name]
        if not (ROOT / rel_path).exists() or "TODO: not written yet" in (ROOT / rel_path).read_text(encoding="utf-8"):
            print(f"[{name}] skipped - {rel_path} not written yet")
            continue

        pipeline = load_pipeline(rel_path)
        print(f"\n[{name}] running {len(questions)} questions with {rel_path}")
        results = []
        for q in questions:
            out = pipeline.run(q["question"])
            results.append({"id": q["id"], "question": q["question"], **out})
            print(f"  {q['id']}  sources={out['sources']}")

        (RESULTS_DIR / out_file).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[{name}] saved {RESULTS_DIR / out_file}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(VERSIONS))
