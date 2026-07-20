import sys

from src.rag.config import RuntimeSettings
from src.rag.storage import delete_stored_file, object_key, store_uploaded_file


def test_object_key_scopes_uploads_by_prefix_and_workspace():
    assert object_key("policy.md", "tenant-a", "docs") == "docs/tenant-a/policy.md"
    assert object_key("policy.md", None, "docs/") == "docs/default/policy.md"


def test_s3_storage_uploads_with_configured_client(tmp_path, monkeypatch):
    uploaded = {}

    class FakeClient:
        def upload_file(self, source, bucket, key):
            uploaded["source"] = source
            uploaded["bucket"] = bucket
            uploaded["key"] = key

    class FakeBoto3:
        @staticmethod
        def client(service, **kwargs):
            uploaded["service"] = service
            uploaded["kwargs"] = kwargs
            return FakeClient()

    path = tmp_path / "policy.md"
    path.write_text("# Policy", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3)

    uri = store_uploaded_file(
        path,
        "policy.md",
        "tenant-a",
        RuntimeSettings(
            object_storage_backend="s3",
            object_storage_bucket="rag-documents",
            object_storage_prefix="docs",
            object_storage_endpoint="https://s3.example",
            object_storage_region="us-east-1",
        ),
    )

    assert uri == "s3://rag-documents/docs/tenant-a/policy.md"
    assert uploaded == {
        "service": "s3",
        "kwargs": {"endpoint_url": "https://s3.example", "region_name": "us-east-1"},
        "source": str(path),
        "bucket": "rag-documents",
        "key": "docs/tenant-a/policy.md",
    }


def test_delete_stored_file_removes_local_file(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text("# Policy", encoding="utf-8")

    assert delete_stored_file(str(path), RuntimeSettings()) is True
    assert not path.exists()


def test_delete_stored_file_removes_s3_object(monkeypatch):
    deleted = {}

    class FakeClient:
        def delete_object(self, Bucket, Key):
            deleted["bucket"] = Bucket
            deleted["key"] = Key

    class FakeBoto3:
        @staticmethod
        def client(service, **kwargs):
            deleted["service"] = service
            deleted["kwargs"] = kwargs
            return FakeClient()

    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3)

    assert delete_stored_file(
        "s3://rag-documents/docs/tenant-a/policy.md",
        RuntimeSettings(object_storage_endpoint="https://s3.example", object_storage_region="us-east-1"),
    ) is True
    assert deleted == {
        "service": "s3",
        "kwargs": {"endpoint_url": "https://s3.example", "region_name": "us-east-1"},
        "bucket": "rag-documents",
        "key": "docs/tenant-a/policy.md",
    }
