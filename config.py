"""
Shared settings for every script in this project.
Keeps model names, paths and the API key in ONE place so each
pipeline file only contains the RAG logic for its topic.

Usage:
    from config import OPENAI_API_KEY, LLM_MODEL, KB_FILE, CHROMA_DIR
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Keep the downloaded cross-encoder model inside the project (D:), not in the C: user cache.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))

# --- API key -----------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and fill it in.")

# --- Models (same for every version, so scores are comparable) ---------------
LLM_MODEL = "gpt-6-luna"           # answers + routing
JUDGE_MODEL = "gpt-6.1-sol"        # LLM-as-judge for evaluation
EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Paths -------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
KB_FILE = DATA_DIR / "dandes_electronics_kb.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"          # git-ignored, never submitted
EVAL_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"

# --- Retrieval constants -----------------------------------------------------
TOP_K = 5
RERANK_CANDIDATES = 20
V1_COLLECTION = "dandes_v1"
IDK = "I don't know based on the available information."

# --- Multi-index layout (Section 5) ------------------------------------------
INDEXES = {
    "policy_index": ["return_policy", "shipping_policy"],
    "product_index": ["product_faq", "known_issue"],
    "order_index": ["order_guide"],
    "ticket_index": ["resolved_ticket"],
}
BM25_DIR = CHROMA_DIR / "bm25"       # pickled BM25 index per collection
