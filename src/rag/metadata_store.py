from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

from src.rag.ingestion import load_manifest, save_manifest


@dataclass
class JsonMetadataStore:
    manifest_path: Path
    jobs_path: Path

    def list_documents(self, workspace_id: str | None = None) -> list[dict]:
        documents = list(load_manifest(self.manifest_path).get("documents", {}).values())
        if workspace_id:
            documents = [document for document in documents if document.get("workspace_id") == workspace_id]
        return sorted(documents, key=lambda document: document.get("ingested_at") or "", reverse=True)

    def get_document(self, document_id: str) -> dict | None:
        document = load_manifest(self.manifest_path).get("documents", {}).get(document_id)
        return dict(document) if document else None

    def save_document(self, document: dict) -> None:
        manifest = load_manifest(self.manifest_path)
        manifest.setdefault("documents", {})[document["document_id"]] = dict(document)
        save_manifest(manifest, self.manifest_path)

    def delete_document(self, document_id: str) -> dict | None:
        manifest = load_manifest(self.manifest_path)
        document = manifest.setdefault("documents", {}).pop(document_id, None)
        save_manifest(manifest, self.manifest_path)
        return dict(document) if document else None

    def list_jobs(self, workspace_id: str | None = None) -> list[dict]:
        jobs = list(_load_jobs(self.jobs_path).values())
        if workspace_id:
            jobs = [job for job in jobs if job.get("workspace_id") == workspace_id]
        return jobs

    def get_job(self, job_id: str) -> dict | None:
        job = _load_jobs(self.jobs_path).get(job_id)
        return dict(job) if job else None

    def upsert_job(self, key: str, **updates) -> dict:
        with _JOB_LOCK:
            jobs = _load_jobs(self.jobs_path)
            job = {**jobs.get(key, {}), **updates}
            jobs[key] = job
            _save_jobs(self.jobs_path, jobs)
            return dict(job)


_JOB_LOCK = threading.Lock()


def _load_jobs(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_jobs(path: Path, jobs: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs, sort_keys=True), encoding="utf-8")
