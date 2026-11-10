import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.rag.api import routes
from src.rag.config import RuntimeSettings


@pytest.fixture
def authorized_client(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(routes, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        routes,
        "SETTINGS",
        RuntimeSettings(auth_mode="api_key", manifest_path=str(manifest)),
    )
    monkeypatch.setattr(
        routes,
        "AUTH_CONTEXTS",
        routes.parse_api_keys(
            "public-key:public:workspace-a,admin-key:admin:workspace-a,"
            "org-key:org-admin:workspace-a,workspace-key:workspace-admin:workspace-a,"
            "other-key:public:workspace-b"
        ),
    )
    return TestClient(routes.create_app()), manifest


@pytest.mark.parametrize(
    "key,status",
    [(None, 401), ("public-key", 403), ("workspace-key", 403), ("org-key", 403)],
)
def test_server_ingestion_requires_global_admin_before_side_effects(
    authorized_client, monkeypatch, key, status
):
    client, _ = authorized_client
    planner = Mock(
        side_effect=AssertionError("Unauthorized ingestion must not touch files")
    )
    monkeypatch.setattr(routes, "plan_document_ingestion", planner)
    response = client.post(
        "/ingest",
        headers={"X-API-Key": key} if key else {},
        json={"pdf_path": "private.pdf"},
    )
    assert response.status_code == status
    planner.assert_not_called()


def test_global_admin_can_use_server_ingestion(authorized_client, monkeypatch):
    client, _ = authorized_client
    planner = Mock(
        return_value=SimpleNamespace(
            should_reindex=False,
            document_id="sample",
            document_version="v1",
            reason="unchanged",
        )
    )
    monkeypatch.setattr(routes, "plan_document_ingestion", planner)
    response = client.post(
        "/ingest", headers={"X-API-Key": "admin-key"}, json={"pdf_path": "sample.pdf"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    planner.assert_called_once()


@pytest.mark.parametrize("roles", [["admin"], "admin"])
def test_source_download_applies_document_roles(authorized_client, tmp_path, roles):
    client, manifest = authorized_client
    source = tmp_path / "restricted.txt"
    source.write_text("Restricted source contents")
    manifest.write_text(
        json.dumps(
            {
                "documents": {
                    "restricted": {
                        "document_id": "restricted",
                        "source_path": str(source),
                        "workspace_id": "workspace-a",
                        "access_roles": roles,
                    }
                }
            }
        )
    )
    public = client.get(
        "/sources/open",
        headers={"X-API-Key": "public-key"},
        params={"path": str(source)},
    )
    assert public.status_code == 404
    assert "Restricted source contents" not in public.text
    admin = client.get(
        "/sources/open",
        headers={"X-API-Key": "admin-key"},
        params={"path": str(source)},
    )
    assert admin.status_code == 200
    assert admin.text == "Restricted source contents"
    cross_workspace = client.get(
        "/sources/open",
        headers={"X-API-Key": "other-key"},
        params={"path": str(source)},
    )
    assert cross_workspace.status_code == 404


def test_legacy_public_source_remains_accessible(authorized_client, tmp_path):
    client, manifest = authorized_client
    source = tmp_path / "public.txt"
    source.write_text("Public source contents")
    manifest.write_text(
        json.dumps(
            {
                "documents": {
                    "public": {
                        "document_id": "public",
                        "source_path": str(source),
                        "workspace_id": "workspace-a",
                    }
                }
            }
        )
    )
    response = client.get(
        "/sources/open",
        headers={"X-API-Key": "public-key"},
        params={"path": str(source)},
    )
    assert response.status_code == 200
