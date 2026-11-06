# Customer Support Copilot — Course 02 Assignment (Dandes Academy)

RAG copilot for the Dandes Electronics support team. It answers questions from the 50-record
knowledge base and cites the source IDs. If the answer is not in the data it replies with
`I don't know based on the available information.`

Three versions, all measured on the same 20-question golden test set (K=5, same models):

- **V1** – basic RAG (dense top 5 → answer) – `v1_basic_rag/2_online_pipeline.py`
- **V2 single index** – multi-query + hybrid search (BM25 + dense, RRF) + metadata boost + cross-encoder rerank + latest-date-wins – `v2_advanced_rag/5_v2_pipeline.py`
- **V2 multi-index + routing** – 4 Chroma indexes, LLM intent router, hybrid search per index, one global rerank – `v2_multi_index/4_v2_multi_index_pipeline.py`

Results: `evaluation/comparison_table.csv`. Demo on the 7 test questions: `demo_output.txt`. Report: `report.pdf`.

## Setup (Windows, PowerShell)

Python 3.12 and pipenv.

```powershell
cd customer_support_copilot
$env:PIPENV_VENV_IN_PROJECT = 1
pipenv install                 # or: pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
copy .env.example .env
```

Put the OpenAI key in `.env` only (`OPENAI_API_KEY=sk-...`). It is git-ignored and never in code.
`config.py` reads it with python-dotenv and holds the model names, paths and constants.
`common.py` has the helpers shared by all scripts (BM25, RRF, reranker, answer generation).
The cross-encoder model (~90 MB) downloads on first use into `.hf_cache/` in the project.

## Run order

From the project root with the venv active.

```powershell
# V1
python v1_basic_rag/1_offline_pipeline.py        # builds Chroma collection dandes_v1, prints 50
python v1_basic_rag/2_online_pipeline.py
python v1_basic_rag/3_conversational_rag.py

# V2 single index
python v2_advanced_rag/1_metadata_filtering.py
python v2_advanced_rag/2_hybrid_search.py
python v2_advanced_rag/3_reranking.py
python v2_advanced_rag/4_query_transformation.py
python v2_advanced_rag/5_v2_pipeline.py

# Multi-index + routing
python v2_multi_index/1_build_indexes.py         # 4 collections + BM25 each, prints 10/20/6/14
python v2_multi_index/2_multi_index_search.py
python v2_multi_index/3_intent_router.py
python v2_multi_index/4_v2_multi_index_pipeline.py --save   # writes demo_output.txt

# Evaluation
python evaluation/1_run_versions.py              # 20 questions x 3 versions -> evaluation/results/
python evaluation/2_retrieval_metrics.py
python evaluation/3_generation_metrics.py        # own judge + RAGAS, takes a few minutes
python evaluation/4_comparison_table.py          # -> evaluation/comparison_table.csv
```

`1_run_versions.py` and `3_generation_metrics.py` take an optional version name (`v1`, `v2_single`, `v2_multi`).

Not submitted / git-ignored: `.env`, `.venv/`, `chroma_db/`, `.hf_cache/`.

## Package versions

Run with langchain 1.4, langchain-chroma 1.1, chromadb 1.5, ragas 0.3.1, sentence-transformers, rank-bm25.
Two compatibility fixes are in the code: gpt-6-luna / gpt-6.1-sol reject a non-default `temperature`,
and ragas imports a module removed in langchain-community 0.4 (stubbed in `evaluation/3_generation_metrics.py`).
