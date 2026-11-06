"""
V1 offline pipeline.
Loads dandes_electronics_kb.json, makes one Document per record, embeds them and stores
them in the Chroma collection 'dandes_v1'. Prints the document count (should be 50).
Run: python v1_basic_rag/1_offline_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config.py / common.py importable

import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import CHROMA_DIR, EMBEDDING_MODEL, KB_FILE, V1_COLLECTION

METADATA_FIELDS = ["product", "brand", "category", "policy_code", "date", "related_policy"]


def load_documents() -> list[Document]:
    """Turn each JSON record into one LangChain Document (one record = one chunk)."""
    with open(KB_FILE, encoding="utf-8") as f:
        records = json.load(f)["documents"]

    docs = []
    for rec in records:
        # title + content is what gets embedded
        text = f"{rec['title']}\n{rec['content']}"

        # Required metadata: id, doc_type + the 6 fields inside rec["metadata"].
        meta = {"id": rec["id"], "doc_type": rec["doc_type"]}
        for field in METADATA_FIELDS:
            value = rec["metadata"].get(field)
            if value is not None:          # Chroma does not accept None values
                meta[field] = value

        docs.append(Document(page_content=text, metadata=meta))
    return docs


def main() -> None:
    docs = load_documents()
    print(f"Loaded {len(docs)} records from {KB_FILE.name}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # delete the old collection so re-running does not duplicate documents
    store = Chroma(
        collection_name=V1_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    store.delete_collection()

    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=V1_COLLECTION,
        persist_directory=str(CHROMA_DIR),
        ids=[d.metadata["id"] for d in docs],   # use FAQ-001 etc. as the Chroma ids
    )

    count = store._collection.count()
    print(f"Stored {count} documents in Chroma collection '{V1_COLLECTION}' at {CHROMA_DIR}")
    assert count == 50, "Expected 50 documents"


if __name__ == "__main__":
    main()
