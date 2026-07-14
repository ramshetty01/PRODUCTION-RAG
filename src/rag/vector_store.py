from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.chunking import DEFAULT_DB_PATH, EMBEDDING_MODEL


def create_embeddings(model_name: str = EMBEDDING_MODEL):
    return HuggingFaceEmbeddings(model_name=model_name)


def build_chroma_db(
    chunks,
    persist_directory: str | Path = DEFAULT_DB_PATH,
    embedding_function=None,
    collection_name: str = "rag_chunks",
):
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)
    embeddings = embedding_function or create_embeddings()
    return Chroma.from_documents(
        documents=list(chunks),
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )


def load_chroma_db(
    persist_directory: str | Path = DEFAULT_DB_PATH,
    embedding_function=None,
    collection_name: str = "rag_chunks",
):
    embeddings = embedding_function or create_embeddings()
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def count_records(vectorstore) -> int:
    return vectorstore._collection.count()
