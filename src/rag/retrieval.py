from __future__ import annotations

from pathlib import Path
import re

from src.rag.advanced.exact_search import exact_search
from src.rag.advanced.sparse_embeddings import sparse_search
from src.rag.citations import citation_id_for_chunk
from src.rag.chunking import DEFAULT_DB_PATH
from src.rag.hybrid_search import hybrid_search
from src.rag.reranking import rerank_chunks
from src.rag.config import RuntimeSettings
from src.rag.vector_store import load_vector_db


DEFAULT_TOP_K = 4
IDENTIFIER_PATTERN = re.compile(r"\b[A-Z]{2,}[-_]\d+|\b[a-z]+[-_][a-z0-9_-]+\b|\b\d{3,}\b")
KEYWORD_LOOKUP_TERMS = {"id", "policy", "control", "owner", "date", "row", "table", "evidence"}


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
        if isinstance(actual, (list, tuple, set)):
            expected_values = set(expected) if isinstance(expected, (list, tuple, set)) else {expected}
            if not set(actual) & expected_values:
                return False
        elif isinstance(expected, (list, tuple, set)):
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


def select_retrieval_strategy(query: str) -> tuple[str, str]:
    query = " ".join(query.split())
    if '"' in query or "'" in query:
        return "exact", "quoted phrase query"
    if IDENTIFIER_PATTERN.search(query):
        return "exact", "identifier-like query"
    lowered = query.lower()
    tokens = set(re.findall(r"\b[a-z0-9_]+\b", lowered))
    if tokens & KEYWORD_LOOKUP_TERMS:
        return "sparse", "keyword/table lookup query"
    if len(query.split()) >= 8:
        return "hybrid", "long conceptual query"
    return "semantic", "short conceptual query"


def load_vectorstore(
    persist_directory: str | Path = DEFAULT_DB_PATH,
    embedding_function=None,
    settings: RuntimeSettings | None = None,
):
    return load_vector_db(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        settings=settings,
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
    exact_documents = [match.document for match in exact_search(query, keyword_documents, top_k=top_k)]
    candidates = hybrid_search(
        query=query,
        vectorstore=vectorstore,
        keyword_documents=keyword_documents,
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
    exact_seen = {citation_id_for_chunk(chunk) for chunk in exact_documents}
    candidates = [*exact_documents, *[chunk for chunk in candidates if citation_id_for_chunk(chunk) not in exact_seen]]
    return filter_authorized_chunks(candidates, metadata_filters=metadata_filters, user_roles=user_roles)[:top_k]


def retrieve_exact_chunks(
    query: str,
    documents,
    top_k: int = DEFAULT_TOP_K,
    metadata_filters: dict | None = None,
    user_roles=None,
):
    matches = exact_search(query, documents, top_k=top_k)
    chunks = [match.document for match in matches]
    return filter_authorized_chunks(chunks, metadata_filters=metadata_filters, user_roles=user_roles)[:top_k]


def retrieve_sparse_chunks(
    query: str,
    documents,
    top_k: int = DEFAULT_TOP_K,
    metadata_filters: dict | None = None,
    user_roles=None,
):
    matches = sparse_search(query, documents, top_k=top_k)
    chunks = [match.document for match in matches]
    return filter_authorized_chunks(chunks, metadata_filters=metadata_filters, user_roles=user_roles)[:top_k]


def retrieve_by_mode(
    query: str,
    mode: str,
    vectorstore=None,
    documents=None,
    top_k: int = DEFAULT_TOP_K,
    reranker=None,
    metadata_filters: dict | None = None,
    user_roles=None,
):
    documents = documents or []
    if mode == "exact":
        return retrieve_exact_chunks(
            query,
            documents,
            top_k=top_k,
            metadata_filters=metadata_filters,
            user_roles=user_roles,
        )
    if mode == "semantic":
        return retrieve_chunks(
            query,
            vectorstore,
            top_k=top_k,
            metadata_filters=metadata_filters,
            user_roles=user_roles,
        )
    if mode == "hybrid":
        return retrieve_hybrid_chunks(
            query,
            vectorstore,
            documents,
            top_k=top_k,
            metadata_filters=metadata_filters,
            user_roles=user_roles,
        )
    if mode == "sparse":
        return retrieve_sparse_chunks(
            query,
            documents,
            top_k=top_k,
            metadata_filters=metadata_filters,
            user_roles=user_roles,
        )
    if mode == "reranked":
        candidates = retrieve_reranked_chunks(query, vectorstore, documents, top_k=top_k, reranker=reranker)
        return filter_authorized_chunks(
            candidates,
            metadata_filters=metadata_filters,
            user_roles=user_roles,
        )[:top_k]
    raise ValueError(f"Unsupported retrieval mode: {mode}")


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
