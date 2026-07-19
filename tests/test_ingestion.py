from langchain_core.documents import Document

from src.rag.config import RuntimeSettings
from src.rag.vector_store import build_chroma_db, build_vector_db, count_records, delete_records_by_metadata, load_chroma_db, load_vector_db


class FakeEmbeddings:
    def _embed(self, text):
        lower = text.lower()
        return [
            float(lower.count("workflow")),
            float(lower.count("runner")),
            float(lower.count("job")),
            float(len(text.split())),
        ]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_build_chroma_db_persists_one_record_per_chunk(tmp_path):
    chunks = [
        Document(
            page_content="A workflow is an automated process.",
            metadata={"source": "/tmp/docs.pdf", "page": 0, "chunk_index": 0, "chunk_id": "docs:p0:c0"},
        ),
        Document(
            page_content="A runner executes jobs.",
            metadata={"source": "/tmp/docs.pdf", "page": 1, "chunk_index": 1, "chunk_id": "docs:p1:c1"},
        ),
    ]
    persist_dir = tmp_path / "chroma_db"
    embeddings = FakeEmbeddings()

    vectorstore = build_chroma_db(chunks, persist_dir, embedding_function=embeddings)
    reloaded = load_chroma_db(persist_dir, embedding_function=embeddings)
    results = reloaded.similarity_search("runner job", k=1)

    assert persist_dir.exists()
    assert count_records(vectorstore) == len(chunks)
    assert count_records(reloaded) == len(chunks)
    assert results[0].metadata["chunk_id"] == "docs:p1:c1"


def test_vector_backend_defaults_to_chroma(tmp_path):
    chunks = [
        Document(
            page_content="A workflow is an automated process.",
            metadata={"source": "/tmp/docs.pdf", "page": 0, "chunk_index": 0, "chunk_id": "docs:p0:c0"},
        )
    ]
    embeddings = FakeEmbeddings()

    vectorstore = build_vector_db(
        chunks,
        tmp_path / "chroma_db",
        embedding_function=embeddings,
        settings=RuntimeSettings(vector_backend="chroma"),
    )
    reloaded = load_vector_db(
        tmp_path / "chroma_db",
        embedding_function=embeddings,
        settings=RuntimeSettings(vector_backend="chroma"),
    )

    assert count_records(vectorstore) == 1
    assert reloaded.similarity_search("workflow", k=1)[0].metadata["chunk_id"] == "docs:p0:c0"


def test_qdrant_backend_uses_managed_adapter(monkeypatch):
    calls = {}

    class FakeQdrantStore:
        collection_name = "prod_chunks"

        def __init__(self, **kwargs):
            calls["load"] = kwargs

        @classmethod
        def from_documents(cls, **kwargs):
            calls["build"] = kwargs
            return cls(collection_name=kwargs["collection_name"])

    monkeypatch.setattr("src.rag.vector_store._require_qdrant", lambda: FakeQdrantStore)
    settings = RuntimeSettings(
        vector_backend="qdrant",
        vector_collection="prod_chunks",
        qdrant_url="https://qdrant.example",
        qdrant_api_key="secret",
    )
    chunks = [Document(page_content="Vendor SOC 2 evidence.", metadata={"chunk_id": "vendor:p0:c0"})]

    build_vector_db(chunks, settings=settings, embedding_function=FakeEmbeddings())
    load_vector_db(settings=settings, embedding_function=FakeEmbeddings())

    assert calls["build"]["collection_name"] == "prod_chunks"
    assert calls["build"]["url"] == "https://qdrant.example"
    assert calls["build"]["api_key"] == "secret"
    assert calls["build"]["documents"][0].metadata["chunk_id"] == "vendor:p0:c0"
    assert calls["load"]["collection_name"] == "prod_chunks"


def test_qdrant_backend_requires_url():
    settings = RuntimeSettings(vector_backend="qdrant")

    try:
        build_vector_db([], settings=settings, embedding_function=FakeEmbeddings())
    except (RuntimeError, ValueError) as exc:
        assert "Qdrant" in str(exc) or "RAG_QDRANT_URL" in str(exc)
    else:
        raise AssertionError("expected qdrant backend to require configuration")


def test_qdrant_delete_uses_tenant_metadata_filter(monkeypatch):
    import sys

    deleted = {}

    class FakeMatchValue:
        def __init__(self, value):
            self.value = value

    class FakeFieldCondition:
        def __init__(self, key, match):
            self.key = key
            self.match = match

    class FakeFilter:
        def __init__(self, must):
            self.must = must

    class FakeModels:
        MatchValue = FakeMatchValue
        FieldCondition = FakeFieldCondition
        Filter = FakeFilter

    class FakeClient:
        def delete(self, collection_name, points_selector):
            deleted["collection_name"] = collection_name
            deleted["points_selector"] = points_selector

    class FakeVectorStore:
        client = FakeClient()
        collection_name = "prod_chunks"

    monkeypatch.setitem(sys.modules, "qdrant_client", type("FakeQdrant", (), {"models": FakeModels}))

    assert delete_records_by_metadata(FakeVectorStore(), {"workspace_id": "tenant-a", "document_id": "policy"}) is None
    conditions = deleted["points_selector"].must
    assert deleted["collection_name"] == "prod_chunks"
    assert [(condition.key, condition.match.value) for condition in conditions] == [
        ("workspace_id", "tenant-a"),
        ("document_id", "policy"),
    ]
