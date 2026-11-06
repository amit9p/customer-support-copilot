"""
Multi-index build.
Splits the 50 records into 4 Chroma collections (policy_index 10, product_index 20,
order_index 6, ticket_index 14) and saves a BM25 index for each one.
Run: python v2_multi_index/1_build_indexes.py
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_chroma import Chroma

from common import BM25Index, get_embeddings, load_module
from config import BM25_DIR, CHROMA_DIR, INDEXES

offline = load_module("v1_basic_rag/1_offline_pipeline.py")     # reuse load_documents()


def main() -> None:
    docs = offline.load_documents()
    embeddings = get_embeddings()
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}

    for name, doc_types in INDEXES.items():
        subset = [d for d in docs if d.metadata["doc_type"] in doc_types]

        # fresh collection each run (no duplicates on re-run)
        Chroma(collection_name=name, embedding_function=embeddings,
               persist_directory=str(CHROMA_DIR)).delete_collection()
        store = Chroma.from_documents(subset, embeddings, collection_name=name,
                                      persist_directory=str(CHROMA_DIR),
                                      ids=[d.metadata["id"] for d in subset])

        with open(BM25_DIR / f"{name}.pkl", "wb") as f:          # keyword index for this collection
            pickle.dump(BM25Index(subset), f)

        counts[name] = store._collection.count()

    print()
    for name, n in counts.items():
        print(f"{name:<14}: {n} documents")
    expected = {"policy_index": 10, "product_index": 20, "order_index": 6, "ticket_index": 14}
    assert counts == expected, f"counts {counts} != expected {expected}"
    print(f"\nBM25 indexes saved to {BM25_DIR}")


if __name__ == "__main__":
    main()
