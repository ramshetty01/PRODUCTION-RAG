from src.rag.metadata_store import JsonMetadataStore


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
