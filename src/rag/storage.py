from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from src.rag.config import RuntimeSettings


def object_key(filename: str, workspace_id: str | None = None, prefix: str = "") -> str:
    parts = [part.strip("/") for part in (prefix, workspace_id or "default", filename) if part and part.strip("/")]
    return "/".join(parts)


def store_uploaded_file(path: str | Path, filename: str, workspace_id: str | None, settings: RuntimeSettings) -> str:
    backend = settings.object_storage_backend.lower()
    if backend == "local":
        return str(Path(path))
    if backend != "s3":
        raise ValueError(f"Unsupported object storage backend: {settings.object_storage_backend}")
    if not settings.object_storage_bucket:
        raise ValueError("RAG_OBJECT_STORAGE_BUCKET is required when RAG_OBJECT_STORAGE_BACKEND=s3")

    try:
        import boto3
    except ImportError as exc:
        raise ValueError("S3 object storage requires the optional boto3 package") from exc

    key = object_key(filename, workspace_id, settings.object_storage_prefix)
    client_kwargs = {}
    if settings.object_storage_endpoint:
        client_kwargs["endpoint_url"] = settings.object_storage_endpoint
    if settings.object_storage_region:
        client_kwargs["region_name"] = settings.object_storage_region
    client = boto3.client("s3", **client_kwargs)
    client.upload_file(str(path), settings.object_storage_bucket, key)
    return f"s3://{settings.object_storage_bucket}/{key}"


def delete_stored_file(uri: str | None, settings: RuntimeSettings) -> bool:
    if not uri:
        return False
    if not uri.startswith("s3://"):
        path = Path(uri)
        if path.exists():
            path.unlink()
            return True
        return False

    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        return False

    import boto3

    client_kwargs = {}
    if settings.object_storage_endpoint:
        client_kwargs["endpoint_url"] = settings.object_storage_endpoint
    if settings.object_storage_region:
        client_kwargs["region_name"] = settings.object_storage_region
    boto3.client("s3", **client_kwargs).delete_object(Bucket=bucket, Key=key)
    return True
