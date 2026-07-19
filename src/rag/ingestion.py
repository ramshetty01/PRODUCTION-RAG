from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_MANIFEST = Path("data/processed/ingestion_manifest.json")


def document_id_for_path(path: str | Path) -> str:
    return Path(path).stem


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict:
    path = Path(path)
    if not path.exists():
        return {"documents": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(manifest: dict, path: str | Path = DEFAULT_MANIFEST) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class IngestionDecision:
    document_id: str
    document_version: str
    content_hash: str
    should_reindex: bool
    reason: str


def plan_document_ingestion(
    source_path: str | Path,
    manifest: dict,
    document_id: str | None = None,
) -> IngestionDecision:
    source_path = Path(source_path)
    document_id = document_id or document_id_for_path(source_path)
    content_hash = file_sha256(source_path)
    existing = manifest.get("documents", {}).get(document_id)

    if existing and existing.get("content_hash") == content_hash:
        return IngestionDecision(
            document_id=document_id,
            document_version=existing["document_version"],
            content_hash=content_hash,
            should_reindex=False,
            reason="unchanged",
        )

    next_version = 1
    if existing:
        next_version = int(str(existing["document_version"]).lstrip("v")) + 1

    return IngestionDecision(
        document_id=document_id,
        document_version=f"v{next_version}",
        content_hash=content_hash,
        should_reindex=True,
        reason="new" if existing is None else "changed",
    )


def record_document_ingestion(
    manifest: dict,
    decision: IngestionDecision,
    source_path: str | Path,
    chunk_count: int,
    status: str = "indexed",
    error: str | None = None,
    storage_uri: str | None = None,
) -> dict:
    documents = manifest.setdefault("documents", {})
    previous = documents.get(decision.document_id)
    documents[decision.document_id] = {
        "document_id": decision.document_id,
        "document_version": decision.document_version,
        "content_hash": decision.content_hash,
        "source_path": str(Path(source_path)),
        "storage_uri": storage_uri or str(Path(source_path)),
        "chunk_count": chunk_count,
        "ingested_at": utc_now_iso(),
        "status": status,
        "previous_version": previous.get("document_version") if previous else None,
        "error": error,
    }
    return manifest
