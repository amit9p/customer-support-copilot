"""
Generation metrics - given the retrieved docs, was the ANSWER good?

    Faithfulness (own judge)  : gpt-6.1-sol splits the answer into claims and checks each one against
                                the retrieved documents. score = supported claims / all claims
    Faithfulness (RAGAS)      : same idea, RAGAS implementation
    Answer Relevancy (RAGAS)  : does the answer actually address the question?
    Context Precision (RAGAS) : are the retrieved docs that matter ranked at the top?
    Context Recall (RAGAS)    : do the retrieved docs cover what the reference answer needs?

Scored on in-scope questions only (an out-of-scope question has no contexts to be faithful to;
refusals are measured separately by Refusal Accuracy).

Run:  python evaluation/3_generation_metrics.py              (all versions with results)
      python evaluation/3_generation_metrics.py v1           (one version)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

# --- compatibility shim -------------------------------------------------------------------
# Bug found: ragas (even 0.4.3) does `from langchain_community.chat_models.vertexai import ChatVertexAI`
# at import time, but langchain-community 0.4 removed that module -> ModuleNotFoundError.
# ragas only uses the class in an isinstance() check, so a stub module is enough.
import types  # noqa: E402

try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
# ------------------------------------------------------------------------------------------

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness  # noqa: E402

from common import get_embeddings  # noqa: E402
from config import EVAL_DIR, IDK, JUDGE_MODEL, KB_FILE, RESULTS_DIR  # noqa: E402

RESULT_FILES = {
    "v1": ("V1 (basic)", "v1_results.json"),
    "v2_single": ("V2 (single index)", "v2_single_index_results.json"),
    "v2_multi": ("V2 (multi-index + routing)", "v2_multi_index_results.json"),
}


class NoTemperatureLLM(LangchainLLMWrapper):
    """RAGAS sets temperature=1e-8 on the model; gpt-6.1-sol only accepts the default. Send none."""

    def get_temperature(self, n: int):
        return None


# --- own LLM-as-judge ---------------------------------------------------------
JUDGE_PROMPT = """You are a strict fact checker. You get the DOCUMENTS a support copilot was given and the
ANSWER it wrote. Split the answer into individual factual claims. For each claim decide whether it is
directly supported by the documents. Return JSON only:
{"claims": [{"claim": "...", "supported": true|false}, ...]}"""


def judge_faithfulness(judge: ChatOpenAI, answer: str, contexts: list[str]) -> float | None:
    msg = f"DOCUMENTS:\n" + "\n\n".join(contexts) + f"\n\nANSWER:\n{answer}"
    text = judge.invoke([SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=msg)]).content
    try:
        claims = json.loads(text.strip().strip("`").removeprefix("json"))["claims"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not claims:
        return None
    return sum(bool(c.get("supported")) for c in claims) / len(claims)


def main(selected: list[str]) -> None:
    kb = {r["id"]: f"[{r['id']}] {r['title']}\n{r['content']}"
          for r in json.loads(KB_FILE.read_text(encoding="utf-8"))["documents"]}
    golden = {q["id"]: q for q in json.loads((EVAL_DIR / "golden_test_set.json").read_text(encoding="utf-8"))["questions"]}

    judge = ChatOpenAI(model=JUDGE_MODEL)
    ragas_llm = NoTemperatureLLM(judge)
    ragas_emb = LangchainEmbeddingsWrapper(get_embeddings())

    all_scores = {}
    for key in selected:
        label, fname = RESULT_FILES[key]
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"{label}: no results file yet")
            continue
        results = [r for r in json.loads(path.read_text(encoding="utf-8")) if golden[r["id"]]["expected_sources"]]
        print(f"\n{label}: scoring {len(results)} in-scope questions")

        # 1) own judge
        own = []
        for r in results:
            contexts = [kb[i] for i in r["retrieved_ids"]]
            score = judge_faithfulness(judge, r["answer"], contexts) if r["answer"] != IDK else 0.0
            if score is not None:
                own.append(score)
            print(f"  {r['id']} own-faithfulness={score}")

        # 2) RAGAS
        samples = [SingleTurnSample(user_input=r["question"], response=r["answer"],
                                    retrieved_contexts=[kb[i] for i in r["retrieved_ids"]],
                                    reference=golden[r["id"]]["reference_answer"]) for r in results]
        ragas_result = evaluate(EvaluationDataset(samples=samples),
                                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                                llm=ragas_llm, embeddings=ragas_emb)
        df = ragas_result.to_pandas()
        (RESULTS_DIR / f"ragas_{key}.csv").write_text(df.to_csv(index=False), encoding="utf-8")

        scores = {
            "Faithfulness (own judge)": round(sum(own) / len(own), 3) if own else None,
            "Faithfulness (RAGAS)": round(float(df["faithfulness"].mean()), 3),
            "Answer Relevancy": round(float(df["answer_relevancy"].mean()), 3),
            "Context Precision": round(float(df["context_precision"].mean()), 3),
            "Context Recall": round(float(df["context_recall"].mean()), 3),
        }
        all_scores[label] = scores
        for k, v in scores.items():
            print(f"  {k:<26} {v}")

    out = RESULTS_DIR / "generation_metrics.json"
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    existing.update(all_scores)
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(RESULT_FILES))
