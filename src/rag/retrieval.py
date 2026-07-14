from __future__ import annotations

from pathlib import Path

from src.rag.chunking import DEFAULT_DB_PATH
from src.rag.vector_store import load_chroma_db


DEFAULT_TOP_K = 4


def load_vectorstore(
    persist_directory: str | Path = DEFAULT_DB_PATH,
    embedding_function=None,
):
    return load_chroma_db(
        persist_directory=str(persist_directory),
        embedding_function=embedding_function,
    )


def retrieve_chunks(query: str, vectorstore, top_k: int = DEFAULT_TOP_K):
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    return vectorstore.similarity_search(query, k=top_k)
