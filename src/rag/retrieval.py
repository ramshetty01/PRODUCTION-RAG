from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.chunking import DEFAULT_DB_PATH, EMBEDDING_MODEL


DEFAULT_TOP_K = 4


def load_vectorstore(
    persist_directory: str | Path = DEFAULT_DB_PATH,
    embedding_model: str = EMBEDDING_MODEL,
):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )


def retrieve_chunks(query: str, vectorstore, top_k: int = DEFAULT_TOP_K):
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    return vectorstore.similarity_search(query, k=top_k)
