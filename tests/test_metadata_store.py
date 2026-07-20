from src.rag.config import RuntimeSettings
from src.rag.metadata_store import JsonMetadataStore, PostgresMetadataStore, SCHEMA_SQL, build_metadata_store


def test_json_metadata_store_manages_documents(tmp_path):
    store = JsonMetadataStore(tmp_path / "manifest.json", tmp_path / "jobs.json")

    store.save_document(
        {
            "document_id": "a",
            "document_version": "v1",
            "workspace_id": "tenant-a",
            "ingested_at": "2026-07-20T00:00:00Z",
        }
    )
    store.save_document(
        {
            "document_id": "b",
            "document_version": "v1",
            "workspace_id": "tenant-b",
            "ingested_at": "2026-07-19T00:00:00Z",
        }
    )

    assert [document["document_id"] for document in store.list_documents("tenant-a")] == ["a"]
    assert store.get_document("a")["workspace_id"] == "tenant-a"
    assert store.delete_document("a")["document_id"] == "a"
    assert store.get_document("a") is None


def test_json_metadata_store_manages_jobs(tmp_path):
    store = JsonMetadataStore(tmp_path / "manifest.json", tmp_path / "jobs.json")

    store.upsert_job("job-a", job_id="job-a", workspace_id="tenant-a", status="queued", progress=0)
    store.upsert_job("job-a", status="indexed", progress=100)
    store.upsert_job("job-b", job_id="job-b", workspace_id="tenant-b", status="queued", progress=0)

    assert store.get_job("job-a")["status"] == "indexed"
    assert [job["job_id"] for job in store.list_jobs("tenant-a")] == ["job-a"]


def test_build_metadata_store_selects_json_or_postgres(tmp_path):
    json_store = build_metadata_store(RuntimeSettings(), tmp_path / "manifest.json", tmp_path / "jobs.json")
    postgres_store = build_metadata_store(
        RuntimeSettings(metadata_backend="postgres", database_url="postgresql://localhost/rag"),
        tmp_path / "manifest.json",
        tmp_path / "jobs.json",
    )

    assert isinstance(json_store, JsonMetadataStore)
    assert isinstance(postgres_store, PostgresMetadataStore)


def test_postgres_metadata_schema_covers_saas_tables():
    for table in ["users", "organizations", "api_keys", "documents", "ingestion_jobs", "chat_sessions", "usage_events"]:
        assert f"create table if not exists {table}" in SCHEMA_SQL


def test_postgres_metadata_store_bootstrap_executes_schema(monkeypatch):
    calls = []

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, params=()):
            calls.append((sql, params))
            return self

        def commit(self):
            calls.append(("commit", ()))

    monkeypatch.setattr(PostgresMetadataStore, "_connect", lambda self: FakeConnection())

    PostgresMetadataStore("postgresql://localhost/rag").bootstrap()

    assert calls[0][0] == SCHEMA_SQL
    assert calls[-1] == ("commit", ())
