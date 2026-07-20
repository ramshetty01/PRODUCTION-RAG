from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

from src.rag.config import RuntimeSettings
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


@dataclass
class PostgresMetadataStore:
    database_url: str

    def bootstrap(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA_SQL)
            conn.commit()

    def list_documents(self, workspace_id: str | None = None) -> list[dict]:
        sql = "select payload from documents"
        params = ()
        if workspace_id:
            sql += " where workspace_id = %s"
            params = (workspace_id,)
        sql += " order by ingested_at desc nulls last"
        with self._connect() as conn:
            return [dict(row[0]) for row in conn.execute(sql, params).fetchall()]

    def get_document(self, document_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("select payload from documents where document_id = %s", (document_id,)).fetchone()
        return dict(row[0]) if row else None

    def save_document(self, document: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into documents (document_id, workspace_id, document_version, ingested_at, payload)
                values (%s, %s, %s, %s, %s)
                on conflict (document_id) do update set
                    workspace_id = excluded.workspace_id,
                    document_version = excluded.document_version,
                    ingested_at = excluded.ingested_at,
                    payload = excluded.payload
                """,
                (
                    document["document_id"],
                    document.get("workspace_id"),
                    document.get("document_version"),
                    document.get("ingested_at"),
                    _json(document),
                ),
            )
            conn.commit()

    def delete_document(self, document_id: str) -> dict | None:
        document = self.get_document(document_id)
        with self._connect() as conn:
            conn.execute("delete from documents where document_id = %s", (document_id,))
            conn.commit()
        return document

    def list_jobs(self, workspace_id: str | None = None) -> list[dict]:
        sql = "select payload from ingestion_jobs"
        params = ()
        if workspace_id:
            sql += " where workspace_id = %s"
            params = (workspace_id,)
        with self._connect() as conn:
            return [dict(row[0]) for row in conn.execute(sql, params).fetchall()]

    def get_job(self, job_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("select payload from ingestion_jobs where job_id = %s", (job_id,)).fetchone()
        return dict(row[0]) if row else None

    def upsert_job(self, key: str, **updates) -> dict:
        job = {**(self.get_job(key) or {}), **updates}
        with self._connect() as conn:
            conn.execute(
                """
                insert into ingestion_jobs (job_id, workspace_id, status, payload)
                values (%s, %s, %s, %s)
                on conflict (job_id) do update set
                    workspace_id = excluded.workspace_id,
                    status = excluded.status,
                    payload = excluded.payload
                """,
                (key, job.get("workspace_id"), job.get("status"), _json(job)),
            )
            conn.commit()
        return job

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url)


def build_metadata_store(settings: RuntimeSettings, manifest_path: Path, jobs_path: Path):
    if settings.metadata_backend == "postgres":
        if not settings.database_url.strip():
            raise ValueError("RAG_DATABASE_URL is required when RAG_METADATA_BACKEND=postgres")
        return PostgresMetadataStore(settings.database_url)
    return JsonMetadataStore(manifest_path, jobs_path)


def _json(value: dict):
    from psycopg.types.json import Jsonb

    return Jsonb(value)


SCHEMA_SQL = """
create table if not exists organizations (
    id text primary key,
    name text not null,
    created_at timestamptz not null default now()
);
create table if not exists users (
    id text primary key,
    organization_id text references organizations(id),
    email text unique not null,
    name text,
    created_at timestamptz not null default now()
);
create table if not exists api_keys (
    id text primary key,
    organization_id text references organizations(id),
    key_hash text unique not null,
    scopes text[] not null default '{}',
    revoked_at timestamptz,
    last_used_at timestamptz,
    created_at timestamptz not null default now()
);
create table if not exists documents (
    document_id text primary key,
    workspace_id text,
    document_version text,
    ingested_at timestamptz,
    payload jsonb not null
);
create table if not exists ingestion_jobs (
    job_id text primary key,
    workspace_id text,
    status text,
    payload jsonb not null
);
create table if not exists chat_sessions (
    id text primary key,
    organization_id text references organizations(id),
    workspace_id text,
    user_id text references users(id),
    created_at timestamptz not null default now()
);
create table if not exists usage_events (
    id bigserial primary key,
    organization_id text references organizations(id),
    workspace_id text,
    user_id text references users(id),
    event_type text not null,
    payload jsonb not null,
    created_at timestamptz not null default now()
);
"""


_JOB_LOCK = threading.Lock()


def _load_jobs(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_jobs(path: Path, jobs: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs, sort_keys=True), encoding="utf-8")
