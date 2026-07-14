from __future__ import annotations

from pathlib import Path

from src.rag.chunking import DEFAULT_DB_PATH
from src.rag.hybrid_search import hybrid_search
from src.rag.reranking import rerank_chunks
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


def retrieve_hybrid_chunks(
    query: str,
    vectorstore,
    keyword_documents,
    top_k: int = DEFAULT_TOP_K,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
):
    return hybrid_search(
        query=query,
        vectorstore=vectorstore,
        keyword_documents=keyword_documents,
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )


def retrieve_reranked_chunks(
    query: str,
    vectorstore,
    keyword_documents,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int | None = None,
    reranker=None,
):
    candidate_k = candidate_k or max(top_k * 3, top_k)
    candidates = retrieve_hybrid_chunks(
        query=query,
        vectorstore=vectorstore,
        keyword_documents=keyword_documents,
        top_k=candidate_k,
    )
    return rerank_chunks(query=query, chunks=candidates, top_k=top_k, reranker=reranker)
