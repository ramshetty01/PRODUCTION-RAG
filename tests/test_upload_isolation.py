import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.rag.api import routes
from src.rag.config import RuntimeSettings


def configure(tmp_path, monkeypatch):
    from src.rag.conversation import ConversationMemoryStore
    from src.rag.performance import QueryCache

    monkeypatch.setattr(routes, "CONVERSATION_MEMORY", ConversationMemoryStore())
    monkeypatch.setattr(routes, "QUERY_CACHE", QueryCache())
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(routes, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        routes,
        "SETTINGS",
        RuntimeSettings(
            auth_mode="api_key",
            manifest_path=str(manifest),
            vector_db_path=str(tmp_path / "vectors"),
        ),
    )
    monkeypatch.setattr(
        routes,
        "AUTH_CONTEXTS",
        routes.parse_api_keys(
            "tenant-a-key:public:workspace-a,tenant-b-key:public:workspace-b"
        ),
    )
    monkeypatch.setattr(
        routes,
        "build_vector_db",
        lambda *args, **kwargs: SimpleNamespace(
            _collection=SimpleNamespace(count=lambda: 1)
        ),
    )
    return TestClient(routes.create_app()), manifest


def test_identical_filenames_have_separate_sources_records_and_chunks(
    tmp_path, monkeypatch
):
    client, manifest = configure(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        routes,
        "build_vector_db",
        lambda chunks, **kwargs: (
            calls.extend(chunks)
            or SimpleNamespace(_collection=SimpleNamespace(count=lambda: 1))
        ),
    )
    results = []
    for key, contents in [
        ("tenant-a-key", b"Workspace A private report."),
        ("tenant-b-key", b"Workspace B different report."),
    ]:
        response = client.post(
            "/upload",
            headers={"X-API-Key": key},
            files={"file": ("report.md", contents, "text/markdown")},
        )
        assert response.status_code == 200, response.text
        results.append(response.json())
    assert results[0]["document_id"] != results[1]["document_id"]
    assert results[0]["saved_path"] != results[1]["saved_path"]
    assert all(result["filename"] == "report.md" for result in results)
    assert Path(results[0]["saved_path"]).read_text() == "Workspace A private report."
    assert Path(results[1]["saved_path"]).read_text() == "Workspace B different report."
    records = json.loads(manifest.read_text())["documents"]
    assert len(records) == 2
    assert {chunk.metadata["document_id"] for chunk in calls} == set(records)
    assert len({chunk.metadata["chunk_id"] for chunk in calls}) == len(calls)
    assert (
        client.get(
            "/sources/open",
            headers={"X-API-Key": "tenant-b-key"},
            params={"path": results[0]["saved_path"]},
        ).status_code
        == 404
    )


def test_concurrent_equal_names_use_independent_quarantine_paths(tmp_path, monkeypatch):
    client, _ = configure(tmp_path, monkeypatch)
    scanned = []
    stored = []
    monkeypatch.setattr(
        routes, "run_upload_scan", lambda path, command: scanned.append(path)
    )
    monkeypatch.setattr(
        routes,
        "store_uploaded_file",
        lambda path, name, workspace, settings: stored.append(name) or str(path),
    )

    def indexed(path, filename, workspace, roles, **kwargs):
        return routes.UploadIngestResponse(
            filename=filename,
            saved_path=str(path),
            document_id=path.stem,
            document_version="v1",
            workspace_id=workspace,
            access_roles=roles,
            status="indexed",
            chunks_created=0,
            vector_records=0,
        )

    monkeypatch.setattr(routes, "_index_uploaded_file", indexed)

    def upload(number):
        return client.post(
            "/upload",
            headers={"X-API-Key": "tenant-a-key"},
            files={"file": ("same.md", str(number).encode(), "text/markdown")},
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(upload, range(4)))
    assert all(response.status_code == 200 for response in responses)
    assert len(set(scanned)) == len(set(stored)) == 4
    assert {
        Path(response.json()["saved_path"]).read_text() for response in responses
    } == {"0", "1", "2", "3"}
    assert not list((tmp_path / "data/quarantine").iterdir())


def test_real_upload_query_download_flow_uses_preserved_display_name(
    tmp_path, monkeypatch
):
    from src.rag.vector_store import build_vector_db

    monkeypatch.setenv("RAG_LLM_PROVIDER", "extractive")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "hash")
    client, _ = configure(tmp_path, monkeypatch)
    monkeypatch.setattr(routes, "build_vector_db", build_vector_db)
    headers = {"X-API-Key": "tenant-a-key"}
    uploaded = client.post(
        "/upload",
        headers=headers,
        files={
            "file": (
                "policy.md",
                b"Cache entries expire after 24 hours.",
                "text/markdown",
            )
        },
    )
    assert uploaded.status_code == 200, uploaded.text
    result = client.post(
        "/query",
        headers=headers,
        json={
            "query": "When do cache entries expire?",
            "persist_dir": str(tmp_path / "vectors"),
        },
    )
    assert result.status_code == 200, result.text
    assert "24 hours" in result.json()["answer"]
    assert result.json()["citations"][0]["source"] == "policy.md"
    source = client.get(
        "/sources/open", headers=headers, params={"path": uploaded.json()["saved_path"]}
    )
    assert source.status_code == 200
    assert source.text == "Cache entries expire after 24 hours."
