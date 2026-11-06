# Customer Support Copilot — Dandes Academy, Course 02 Assignment

A RAG copilot for the Dandes Electronics support team. A support executive asks a question;
the copilot answers from the 50-record knowledge base and cites the source IDs it used.
If the answer is not in the data it replies exactly: `I don't know based on the available information.`

Three versions are built and measured on the same 20-question golden test set:

| Version | What it is | Entry point |
|---|---|---|
| V1 | basic RAG: dense top-5 → answer | `v1_basic_rag/2_online_pipeline.py` |
| V2 (single index) | multi-query + hybrid (BM25+dense, RRF) + metadata boost + cross-encoder rerank + "latest date wins" | `v2_advanced_rag/5_v2_pipeline.py` |
| V2 (multi-index + routing) | 4 Chroma indexes, LLM intent router, hybrid per index, one global rerank | `v2_multi_index/4_v2_multi_index_pipeline.py` |

## 1. Setup (Windows, PowerShell)

Requires Python 3.12 and [pipenv](https://pipenv.pypa.io/).

```powershell
cd customer_support_copilot
$env:PIPENV_VENV_IN_PROJECT = 1
pipenv install                 # creates .venv from Pipfile (or: pip install -r requirements.txt)
.\.venv\Scripts\Activate.ps1
copy .env.example .env         # then put your key in .env
```

### Where the OpenAI key goes
Only in `.env` (git-ignored). Never in code.

```
OPENAI_API_KEY=sk-...
```

`config.py` loads it with `python-dotenv`. All model names, paths and constants (K=5, collection
names, the fixed "I don't know" sentence) live in `config.py`; shared helpers (BM25, RRF, reranker,
answer generation) live in `common.py`. The cross-encoder model (~90 MB) downloads on first use into
`.hf_cache/` inside the project.

## 2. Run order

Run everything from the project root with the venv active. Each script prints what it did.

```powershell
# --- V1: basic RAG ---------------------------------------------------------
python v1_basic_rag/1_offline_pipeline.py        # builds Chroma collection dandes_v1 (prints 50)
python v1_basic_rag/2_online_pipeline.py         # 3 demo questions with sources
python v1_basic_rag/3_conversational_rag.py      # follow-up questions

# --- V2: advanced RAG (single index) --------------------------------------
python v2_advanced_rag/1_metadata_filtering.py   # before / after demo
python v2_advanced_rag/2_hybrid_search.py        # before / after demo
python v2_advanced_rag/3_reranking.py            # before / after demo
python v2_advanced_rag/4_query_transformation.py # before / after for (A) rewrite and (B) multi-query
python v2_advanced_rag/5_v2_pipeline.py          # all four combined, 7 test questions

# --- Multi-index + routing -------------------------------------------------
python v2_multi_index/1_build_indexes.py         # 4 collections + BM25 each (prints 10/20/6/14)
python v2_multi_index/2_multi_index_search.py    # all-4 vs order+ticket demo
python v2_multi_index/3_intent_router.py         # router test on the 7 questions
python v2_multi_index/4_v2_multi_index_pipeline.py --save   # 7 questions, writes demo_output.txt

# --- Evaluation ------------------------------------------------------------
python evaluation/1_run_versions.py              # 20 questions x 3 versions -> evaluation/results/*.json
python evaluation/2_retrieval_metrics.py         # Precision/Recall/Coverage/Routing/Refusals
python evaluation/3_generation_metrics.py        # own LLM judge + RAGAS (slow: ~5 min, many LLM calls)
python evaluation/4_comparison_table.py          # -> evaluation/comparison_table.csv
```

`1_run_versions.py` and `3_generation_metrics.py` accept version names (`v1`, `v2_single`, `v2_multi`)
to run one version at a time.

## 3. Layout

```
customer_support_copilot/
├── config.py, common.py        shared settings and helpers (see above)
├── data/dandes_electronics_kb.json
├── v1_basic_rag/               3 files
├── v2_advanced_rag/            5 files
├── v2_multi_index/             4 files
├── evaluation/                 golden_test_set.json, 4 scripts, results/, comparison_table.csv
├── demo_output.txt             final pipeline on the 7 test questions
├── report.pdf
└── requirements.txt, Pipfile
```

Not submitted / git-ignored: `.env`, `.venv/`, `chroma_db/`, `.hf_cache/`.

## 4. Notes on versions used

Tested with langchain 1.4, langchain-chroma 1.1, chromadb 1.5, ragas 0.3.1, sentence-transformers,
rank-bm25. Two compatibility fixes are in the code and explained in the report:
`gpt-6-luna` / `gpt-6.1-sol` reject a non-default `temperature`, and ragas imports a module that
langchain-community 0.4 removed (stubbed in `evaluation/3_generation_metrics.py`).
