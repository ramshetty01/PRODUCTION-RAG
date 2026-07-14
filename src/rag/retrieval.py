from __future__ import annotations

from pathlib import Path

from src.rag.chunking import DEFAULT_DB_PATH
from src.rag.hybrid_search import hybrid_search
from src.rag.reranking import rerank_chunks
from src.rag.vector_store import load_chroma_db


DEFAULT_TOP_K = 4


def user_can_access(chunk, user_roles: set[str] | None = None) -> bool:
    allowed_roles = chunk.metadata.get("access_roles") or ["public"]
    if isinstance(allowed_roles, str):
        allowed_roles = [allowed_roles]
    allowed_roles = set(allowed_roles)
    user_roles = user_roles or {"public"}
    return "public" in allowed_roles or bool(allowed_roles & user_roles)


def metadata_matches(chunk, filters: dict | None = None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = chunk.metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def filter_authorized_chunks(chunks, metadata_filters: dict | None = None, user_roles=None):
    roles = set(user_roles or {"public"})
    return [
        chunk
        for chunk in chunks
        if user_can_access(chunk, roles) and metadata_matches(chunk, metadata_filters)
    ]


def load_vectorstore(
    persist_directory: str | Path = DEFAULT_DB_PATH,
    embedding_function=None,
):
    return load_chroma_db(
        persist_directory=str(persist_directory),
        embedding_function=embedding_function,
    )


def retrieve_chunks(
    query: str,
    vectorstore,
    top_k: int = DEFAULT_TOP_K,
    metadata_filters: dict | None = None,
    user_roles=None,
):
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    candidates = vectorstore.similarity_search(query, k=top_k)
    return filter_authorized_chunks(candidates, metadata_filters=metadata_filters, user_roles=user_roles)[:top_k]


def retrieve_hybrid_chunks(
    query: str,
    vectorstore,
    keyword_documents,
    top_k: int = DEFAULT_TOP_K,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
    metadata_filters: dict | None = None,
    user_roles=None,
):
    candidates = hybrid_search(
        query=query,
        vectorstore=vectorstore,
        keyword_documents=keyword_documents,
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
    return filter_authorized_chunks(candidates, metadata_filters=metadata_filters, user_roles=user_roles)[:top_k]


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
