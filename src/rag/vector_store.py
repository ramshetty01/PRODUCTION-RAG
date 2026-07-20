from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from src.rag.chunking import DEFAULT_DB_PATH
from src.rag.config import DEFAULT_EMBEDDING_MODEL, RuntimeSettings
from src.rag.models import get_model_provider

SUPPORTED_VECTOR_BACKENDS = {"chroma", "qdrant"}


def create_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL, settings: RuntimeSettings | None = None):
    if settings is not None:
        return get_model_provider(settings).embeddings()
    settings = RuntimeSettings(embedding_model=model_name)
    return get_model_provider(settings).embeddings()


def build_chroma_db(
    chunks,
    persist_directory: str | Path = DEFAULT_DB_PATH,
    embedding_function=None,
    collection_name: str = "rag_chunks",
    settings: RuntimeSettings | None = None,
):
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)
    embeddings = embedding_function or create_embeddings(settings=settings)
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
    settings: RuntimeSettings | None = None,
):
    embeddings = embedding_function or create_embeddings(settings=settings)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def _require_qdrant():
    try:
        from langchain_qdrant import QdrantVectorStore
    except ImportError as exc:
        raise RuntimeError(
            "Qdrant backend requires the optional langchain-qdrant package. "
            "Install it and set RAG_VECTOR_BACKEND=qdrant with RAG_QDRANT_URL."
        ) from exc
    return QdrantVectorStore


def _qdrant_connection(settings: RuntimeSettings) -> dict:
    if not settings.qdrant_url:
        raise ValueError("RAG_QDRANT_URL is required when RAG_VECTOR_BACKEND=qdrant")
    connection = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        connection["api_key"] = settings.qdrant_api_key
    return connection


def build_vector_db(
    chunks,
    persist_directory: str | Path = DEFAULT_DB_PATH,
    embedding_function=None,
    settings: RuntimeSettings | None = None,
):
    settings = settings or RuntimeSettings()
    backend = settings.vector_backend.lower()
    if backend not in SUPPORTED_VECTOR_BACKENDS:
        raise ValueError(f"Unsupported vector backend: {settings.vector_backend}")
    if backend == "chroma":
        return build_chroma_db(
            chunks,
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=settings.vector_collection,
            settings=settings,
        )

    qdrant = _require_qdrant()
    embeddings = embedding_function or create_embeddings(settings=settings)
    return qdrant.from_documents(
        documents=list(chunks),
        embedding=embeddings,
        collection_name=settings.vector_collection,
        **_qdrant_connection(settings),
    )


def load_vector_db(
    persist_directory: str | Path = DEFAULT_DB_PATH,
    embedding_function=None,
    settings: RuntimeSettings | None = None,
):
    settings = settings or RuntimeSettings()
    backend = settings.vector_backend.lower()
    if backend not in SUPPORTED_VECTOR_BACKENDS:
        raise ValueError(f"Unsupported vector backend: {settings.vector_backend}")
    if backend == "chroma":
        return load_chroma_db(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=settings.vector_collection,
            settings=settings,
        )

    qdrant = _require_qdrant()
    embeddings = embedding_function or create_embeddings(settings=settings)
    return qdrant(
        embedding=embeddings,
        collection_name=settings.vector_collection,
        **_qdrant_connection(settings),
    )


def count_records(vectorstore) -> int:
    if hasattr(vectorstore, "_collection"):
        return vectorstore._collection.count()
    client = getattr(vectorstore, "client", None)
    collection_name = getattr(vectorstore, "collection_name", None) or getattr(vectorstore, "_collection_name", None)
    if client is not None and collection_name:
        return int(client.count(collection_name=collection_name, exact=True).count)
    raise TypeError("vector store does not expose a supported record count API")


def delete_records_by_metadata(vectorstore, metadata_filter: dict) -> int | None:
    if not hasattr(vectorstore, "_collection"):
        client = getattr(vectorstore, "client", None)
        collection_name = getattr(vectorstore, "collection_name", None) or getattr(vectorstore, "_collection_name", None)
        if client is not None and collection_name:
            try:
                from qdrant_client import models
            except ImportError as exc:
                raise TypeError("Qdrant metadata delete requires qdrant-client") from exc
            points_filter = models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in metadata_filter.items()
                ]
            )
            client.delete(collection_name=collection_name, points_selector=points_filter)
            return None
        raise TypeError("vector store does not expose a supported metadata delete API")
    before = count_records(vectorstore)
    vectorstore._collection.delete(where=metadata_filter)
    after = count_records(vectorstore)
    return before - after
