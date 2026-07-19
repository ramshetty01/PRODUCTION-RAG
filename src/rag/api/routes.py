from __future__ import annotations

import html
import json
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.rag.auth import AuthContext, authenticate_request, parse_api_keys
from src.rag.audit import DEFAULT_AUDIT_LOG, append_audit_event, audit_events_csv, load_audit_events
from src.rag.chunking import (
    DEFAULT_DB_PATH,
    DEFAULT_PDF_PATH,
    SUPPORTED_DOCUMENT_SUFFIXES,
    chunk_file,
    chunk_pdf,
    chunk_token_summary,
    count_tokens,
)
from src.rag.config import load_settings
from src.rag.conversation import ConversationMemoryStore, ConversationTurn, build_contextual_query
from src.rag.evaluation_report import build_evaluation_report
from src.rag.generation import generate_answer
from src.rag.ingestion import DEFAULT_MANIFEST, load_manifest, plan_document_ingestion, record_document_ingestion, save_manifest
from src.rag.monitoring import DEFAULT_FEEDBACK_LOG, FeedbackEvent, append_feedback, load_feedback, monitoring_metrics
from src.rag.models import LLM_PROVIDER_FAILURES, get_model_provider
from src.rag.observability import (
    LOGGER,
    MetricsRegistry,
    TraceEvent,
    chunk_ids,
    citation_ids,
    configure_managed_observability,
    configure_opentelemetry,
    new_request_id,
    structured_request_log,
    trace_latency,
)
from src.rag.performance import build_query_cache, estimate_llm_cost
from src.rag.citations import productize_answer_citations, productize_citations
from src.rag.reranking import build_reranker
from src.rag.retrieval import DEFAULT_TOP_K, load_vectorstore, retrieve_by_mode, select_retrieval_strategy
from src.rag.runtime_quality import score_runtime_answer
from src.rag.security import build_rate_limiter, run_upload_scan, sanitize_upload_filename, validate_path, validate_query
from src.rag.usage import DEFAULT_USAGE_LOG, UsageEvent, append_usage, load_usage, usage_summary
from src.rag.vector_store import build_vector_db, count_records, delete_records_by_metadata


SETTINGS = load_settings()
SUPPORTED_RETRIEVAL_MODES = {"auto", "semantic", "exact", "hybrid", "sparse", "reranked"}
AUTH_CONTEXTS = parse_api_keys(SETTINGS.api_keys)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = PROJECT_ROOT / "demo"
LEGAL_DIR = PROJECT_ROOT / "docs" / "legal"
SUPPORTED_UPLOAD_SUFFIXES = SUPPORTED_DOCUMENT_SUFFIXES
LEGAL_PAGES = {
    "privacy": "privacy.md",
    "terms": "terms.md",
    "data-deletion": "data-deletion.md",
    "subprocessors": "subprocessors.md",
}


class HealthResponse(BaseModel):
    status: str


class IngestRequest(BaseModel):
    pdf_path: str = Field(default=str(DEFAULT_PDF_PATH))
    persist_dir: str = Field(default=SETTINGS.vector_db_path)
    manifest_path: str = Field(default=SETTINGS.manifest_path)
    build_vector_db: bool = True


class IngestResponse(BaseModel):
    document_id: str
    document_version: str
    status: str
    reason: str
    chunks_created: int
    min_tokens: int | None
    max_tokens: int | None
    vector_records: int | None
    chunk_summary: dict | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=SETTINGS.top_k, gt=0)
    retrieval_mode: str = Field(default=SETTINGS.retrieval_mode)
    persist_dir: str = Field(default=SETTINGS.vector_db_path)
    workspace_id: str | None = None
    session_id: str | None = None
    metadata_filters: dict | None = None
    user_roles: list[str] = Field(default_factory=lambda: ["public"], deprecated=True)


class CitationResponse(BaseModel):
    id: str
    label: str
    source: str
    source_path: str
    page: int | None
    chunk_index: int | None
    snippet: str
    context: str
    source_url: str
    quote: str


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[CitationResponse]
    quality: dict
    retrieval: dict
    trace: dict
    cached: bool = False


class FeedbackRequest(BaseModel):
    request_id: str
    query: str
    answer: str
    helpful: bool
    citations: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
    note: str | None = None
    feedback_path: str = Field(default=str(DEFAULT_FEEDBACK_LOG))


class FeedbackResponse(BaseModel):
    status: str
    request_id: str


class FeedbackEventsResponse(BaseModel):
    events: list[dict]


class MonitoringResponse(BaseModel):
    metrics: dict


class UsageResponse(BaseModel):
    usage: dict


class LLMHealthResponse(BaseModel):
    status: str
    provider: str
    model: str | None = None
    endpoint: str | None = None
    error: str | None = None


class EvaluationResponse(BaseModel):
    generated_at: str
    dataset: dict
    config: dict
    metrics: dict
    quality_gate: dict
    case_scores: list[dict]


class UploadIngestResponse(BaseModel):
    filename: str
    saved_path: str
    document_id: str
    document_version: str
    workspace_id: str = "default"
    access_roles: list[str] = Field(default_factory=lambda: ["public"])
    status: str
    job_id: str | None = None
    progress: int | None = None
    chunks_created: int
    vector_records: int | None
    chunk_summary: dict | None = None


class IngestionJobResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    document_id: str | None = None
    document_version: str | None = None
    chunks_created: int = 0
    vector_records: int | None = None
    error: str | None = None


class DocumentRecordResponse(BaseModel):
    document_id: str
    document_version: str
    filename: str | None = None
    workspace_id: str | None = None
    owner: str | None = None
    status: str
    chunk_count: int
    source_path: str
    ingested_at: str | None = None
    error: str | None = None
    retry_action: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecordResponse]


class IndexReadinessResponse(BaseModel):
    status: str
    ready: bool
    message: str
    document_count: int
    pending_jobs: int = 0
    failed_jobs: int = 0


class DeleteDocumentResponse(BaseModel):
    document_id: str
    status: str
    vector_records_deleted: int | None = None


class RenameDocumentRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=160)


class PurgeWorkspaceResponse(BaseModel):
    workspace_id: str
    documents_deleted: int
    files_deleted: int
    vector_records_deleted: int | None
    conversations_deleted: int
    logs_deleted: int


class RetentionRunResponse(BaseModel):
    status: str
    documents_deleted: int
    files_deleted: int
    vector_records_deleted: int | None


class AdminStatusResponse(BaseModel):
    health: dict
    index: dict
    documents: list[DocumentRecordResponse]
    failed_jobs: list[dict]


class ObservabilityDashboardResponse(BaseModel):
    window: dict
    metrics: dict
    request_latency: dict
    retrieval: dict
    ingestion: dict
    model: dict
    index_health: dict
    feedback: dict
    recent_events: dict


class AuditResponse(BaseModel):
    events: list[dict]


router = APIRouter()
QUERY_CACHE = build_query_cache(SETTINGS.cache_backend, SETTINGS.redis_url)
RATE_LIMITER = build_rate_limiter(SETTINGS.rate_limit_backend, SETTINGS.redis_url, max_requests=120, window_seconds=60)
METRICS = MetricsRegistry()
OTEL = configure_opentelemetry(SETTINGS)
MANAGED_OBSERVABILITY = configure_managed_observability(SETTINGS)
CONVERSATION_MEMORY = ConversationMemoryStore(max_turns=SETTINGS.conversation_max_turns)
INGESTION_JOBS: dict[str, dict] = {}
INGESTION_JOB_LOCK = threading.Lock()
MODEL_ERROR_COUNT = 0
RETENTION_THREAD_STARTED = False


def _auth_context(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> AuthContext:
    try:
        return authenticate_request(
            SETTINGS.auth_mode,
            api_key=x_api_key,
            authorization_header=authorization,
            configured_keys=AUTH_CONTEXTS,
            jwt_secret=SETTINGS.jwt_secret,
            jwt_issuer=SETTINGS.jwt_issuer,
            jwt_audience=SETTINGS.jwt_audience,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _admin_context(auth_context: AuthContext = Depends(_auth_context)) -> AuthContext:
    if not (auth_context.roles & {"admin", "org-admin", "workspace-admin"}):
        raise HTTPException(status_code=403, detail="admin role required")
    return auth_context


def _require_global_admin(auth_context: AuthContext) -> None:
    if not (auth_context.roles & {"admin", "org-admin"}):
        raise HTTPException(status_code=403, detail="org admin role required")


def _require_workspace_admin(auth_context: AuthContext, workspace_id: str | None) -> None:
    if auth_context.roles & {"admin", "org-admin"}:
        return
    if "workspace-admin" in auth_context.roles and workspace_id and auth_context.tenant_id == workspace_id:
        return
    raise HTTPException(status_code=403, detail="workspace admin role required")


def _audit_admin_action(action: str, auth_context: AuthContext, workspace_id: str | None, target: str | None = None) -> None:
    append_audit_event(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": action,
            "subject": auth_context.subject,
            "tenant_id": auth_context.tenant_id,
            "workspace_id": workspace_id or "default",
            "target": target,
        },
        _safe_api_path(DEFAULT_AUDIT_LOG),
    )


def _safe_api_path(path: str | Path) -> Path:
    try:
        return validate_path(path, PROJECT_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _candidate_documents(vectorstore, query: str, top_k: int):
    return vectorstore.similarity_search(query, k=max(top_k * 4, top_k))


def _clear_query_cache() -> None:
    if hasattr(QUERY_CACHE, "values"):
        QUERY_CACHE.values.clear()


def _product_error_detail(
    code: str,
    message: str,
    *,
    retry: str,
    actions: list[str] | None = None,
    request_id: str | None = None,
    trace: dict | None = None,
) -> dict:
    detail = {
        "code": code,
        "message": message,
        "retry": retry,
        "actions": actions or ["retry"],
    }
    if request_id:
        detail["request_id"] = request_id
    if trace:
        detail["trace"] = trace
    return detail


def _classify_query_error(exc: Exception, request_id: str, event: TraceEvent) -> dict:
    global MODEL_ERROR_COUNT
    text = str(exc).lower()
    event.error = type(exc).__name__
    trace = event.to_log_dict()
    if any(term in text for term in ["vector", "chroma", "collection", "persist", "index"]):
        return _product_error_detail(
            "vector_store_unavailable",
            "The document index is not available right now.",
            retry="Reindex the corpus, then retry the question.",
            actions=["reindex", "retry", "contact_admin"],
            request_id=request_id,
            trace=trace,
        )
    if any(term in text for term in ["retriev", "search"]):
        return _product_error_detail(
            "retrieval_unavailable",
            "We could not search the indexed corpus right now.",
            retry="Retry the question. If it repeats, reindex the corpus.",
            actions=["retry", "reindex"],
            request_id=request_id,
            trace=trace,
        )
    if any(term in text for term in ["llm", "model", "openrouter", "completion", "chat"]):
        MODEL_ERROR_COUNT += 1
        return _product_error_detail(
            "answer_generation_unavailable",
            "The answer model did not respond.",
            retry="Retry the question or check the configured LLM provider.",
            actions=["retry", "contact_admin"],
            request_id=request_id,
            trace=trace,
        )
    return _product_error_detail(
        "rag_request_failed",
        "We could not answer this question right now.",
        retry="Retry the request. If it repeats, check the request trace.",
        actions=["retry", "contact_admin"],
        request_id=request_id,
        trace=trace,
    )


def _classify_ingestion_error(exc: Exception) -> dict:
    text = str(exc).lower()
    if any(term in text for term in ["parse", "loader", "pdf", "docx", "pptx", "decode"]):
        return _product_error_detail(
            "document_parse_failed",
            "We could not read this file.",
            retry="Upload a clean PDF, Markdown, text, HTML, CSV, DOCX, or PPTX export.",
            actions=["reupload"],
        )
    if any(term in text for term in ["embedding", "embed"]):
        return _product_error_detail(
            "embedding_failed",
            "We could not create embeddings for this document.",
            retry="Retry indexing after checking the embedding provider.",
            actions=["reindex", "contact_admin"],
        )
    if any(term in text for term in ["vector", "chroma", "collection", "persist", "index"]):
        return _product_error_detail(
            "index_write_failed",
            "We could not write this document to the search index.",
            retry="Retry indexing. If it repeats, reset or inspect the vector database.",
            actions=["reindex", "contact_admin"],
        )
    return _product_error_detail(
        "indexing_failed",
        "We could not index this document.",
        retry="Retry with a smaller or cleaner document.",
        actions=["reupload", "contact_admin"],
    )


def _set_ingestion_job(job_key: str, **updates) -> None:
    with INGESTION_JOB_LOCK:
        INGESTION_JOBS[job_key] = {**INGESTION_JOBS.get(job_key, {}), **updates}


def _get_ingestion_job(job_id: str) -> dict | None:
    with INGESTION_JOB_LOCK:
        job = INGESTION_JOBS.get(job_id)
        return dict(job) if job else None


def _ingestion_jobs_for_workspace(workspace_id: str | None) -> list[dict]:
    with INGESTION_JOB_LOCK:
        jobs = [dict(job) for job in INGESTION_JOBS.values()]
    if workspace_id:
        return [job for job in jobs if job.get("workspace_id") == workspace_id]
    return jobs


def _index_readiness(workspace_id: str | None = None) -> IndexReadinessResponse:
    safe_workspace_id = _safe_workspace_id(workspace_id)
    documents = _document_records(safe_workspace_id)
    jobs = _ingestion_jobs_for_workspace(safe_workspace_id)
    pending_jobs = [job for job in jobs if job.get("status") in {"queued", "parsing", "chunking", "embedding"}]
    failed_jobs = [job for job in jobs if job.get("status") == "failed"]
    indexed_documents = [document for document in documents if document.get("status") in {"indexed", "skipped"}]

    if failed_jobs:
        return IndexReadinessResponse(
            status="failed",
            ready=False,
            message="Indexing failed. Upload the corpus again.",
            document_count=len(indexed_documents),
            pending_jobs=len(pending_jobs),
            failed_jobs=len(failed_jobs),
        )
    if pending_jobs:
        return IndexReadinessResponse(
            status="indexing",
            ready=False,
            message="Indexing corpus before chat is enabled.",
            document_count=len(indexed_documents),
            pending_jobs=len(pending_jobs),
        )
    if indexed_documents:
        return IndexReadinessResponse(
            status="ready",
            ready=True,
            message="Corpus indexed. You can ask questions.",
            document_count=len(indexed_documents),
        )
    return IndexReadinessResponse(
        status="empty",
        ready=False,
        message="Upload and index a corpus before asking.",
        document_count=0,
    )


def _parse_event_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _filter_window(events: list, key: str, since: datetime) -> list:
    filtered = []
    for event in events:
        value = getattr(event, key, None) if not isinstance(event, dict) else event.get(key)
        parsed = _parse_event_time(value)
        if parsed is None or parsed >= since:
            filtered.append(event)
    return filtered


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return round(ordered[index], 3)


def _observability_dashboard(window_minutes: int, workspace_id: str | None = None) -> ObservabilityDashboardResponse:
    safe_window_minutes = max(1, min(window_minutes, 24 * 60))
    since = datetime.now(UTC) - timedelta(minutes=safe_window_minutes)
    audit_events = _filter_window(load_audit_events(_safe_api_path(DEFAULT_AUDIT_LOG), limit=1000), "timestamp", since)
    feedback_events = _filter_window(load_feedback(_safe_api_path(DEFAULT_FEEDBACK_LOG)), "created_at", since)
    latency_values = [float(event.get("latency_ms")) for event in audit_events if event.get("latency_ms") is not None]
    retrieval_counts = [len(event.get("retrieval_ids") or []) for event in audit_events]
    documents = _document_records(workspace_id)
    jobs = _ingestion_jobs_for_workspace(_safe_workspace_id(workspace_id))
    failed_jobs = [job for job in jobs if job.get("status") == "failed"]
    pending_jobs = [job for job in jobs if job.get("status") in {"queued", "parsing", "chunking", "embedding"}]
    failed_documents = [document for document in documents if document.get("status") not in {"indexed", "skipped"}]
    request_count = METRICS.request_count
    return ObservabilityDashboardResponse(
        window={"minutes": safe_window_minutes, "since": since.isoformat()},
        metrics={
            "request_count": request_count,
            "status_counts": {str(code): count for code, count in METRICS.status_counts.items()},
            "avg_latency_ms": round(METRICS.latency_ms_total / request_count, 3) if request_count else 0.0,
        },
        request_latency={
            "count": len(latency_values),
            "avg_ms": round(sum(latency_values) / len(latency_values), 3) if latency_values else 0.0,
            "p95_ms": _percentile(latency_values, 0.95),
            "max_ms": max(latency_values) if latency_values else 0.0,
        },
        retrieval={
            "requests": len(retrieval_counts),
            "total_chunks": sum(retrieval_counts),
            "avg_chunks": round(sum(retrieval_counts) / len(retrieval_counts), 3) if retrieval_counts else 0.0,
        },
        ingestion={
            "pending_jobs": len(pending_jobs),
            "failed_jobs": len(failed_jobs),
            "failed_documents": len(failed_documents),
        },
        model={"errors": MODEL_ERROR_COUNT},
        index_health=_index_readiness(workspace_id).model_dump(),
        feedback=monitoring_metrics(feedback_events),
        recent_events={
            "audit": audit_events[:10],
            "feedback": [event.__dict__ for event in list(reversed(feedback_events))[:10]],
            "failed_jobs": failed_jobs[:10],
        },
    )


def _chunks_for_path(path: Path, document_version: str):
    return chunk_file(
        path,
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
        document_version=document_version,
    )


def _safe_workspace_id(workspace_id: str | None) -> str | None:
    if workspace_id is None or workspace_id.strip() == "":
        return None
    value = workspace_id.strip()
    if len(value) > 80:
        raise HTTPException(status_code=400, detail="workspace_id is too long")
    if any(not (character.isalnum() or character in {"-", "_"}) for character in value):
        raise HTTPException(
            status_code=400,
            detail="workspace_id may only contain letters, numbers, hyphen, and underscore",
        )
    return value


def _safe_session_id(session_id: str | None) -> str | None:
    if session_id is None or session_id.strip() == "":
        return None
    value = session_id.strip()
    if len(value) > 120:
        raise HTTPException(status_code=400, detail="session_id is too long")
    if any(not (character.isalnum() or character in {"-", "_"}) for character in value):
        raise HTTPException(
            status_code=400,
            detail="session_id may only contain letters, numbers, hyphen, and underscore",
        )
    return value


def _safe_access_roles(access_roles: str | None) -> list[str]:
    if access_roles is None or access_roles.strip() == "":
        return ["public"]
    roles = [role.strip() for role in re.split(r"[,|]", access_roles) if role.strip()]
    if not roles:
        return ["public"]
    for role in roles:
        if len(role) > 64 or not re.match(r"^[A-Za-z0-9_-]+$", role):
            raise HTTPException(status_code=400, detail="access_roles may only contain role names separated by comma or pipe")
    return sorted(set(roles))


def _effective_metadata_filters(request: QueryRequest) -> dict:
    metadata_filters = dict(request.metadata_filters or {})
    workspace_id = _safe_workspace_id(request.workspace_id)
    if workspace_id:
        metadata_filters["workspace_id"] = workspace_id
    return metadata_filters


def _document_manifest_path() -> Path:
    return _safe_api_path(SETTINGS.manifest_path)


def _document_records(workspace_id: str | None = None) -> list[dict]:
    safe_workspace_id = _safe_workspace_id(workspace_id)
    manifest = load_manifest(_document_manifest_path())
    records = []
    for record in manifest.get("documents", {}).values():
        records.append(_document_response_record(record))
    if safe_workspace_id:
        records = [record for record in records if record.get("workspace_id") == safe_workspace_id]
    return sorted(records, key=lambda record: record.get("ingested_at") or "", reverse=True)


def _document_response_record(record: dict) -> dict:
    row = dict(record)
    row["owner"] = row.get("owner") or row.get("uploaded_by") or "unknown"
    row["retry_action"] = "reindex" if row.get("status") not in {"indexed", "skipped"} else None
    return row


def _index_uploaded_file(
    saved_path: Path,
    safe_name: str,
    safe_workspace_id: str | None,
    safe_access_roles: list[str],
    owner: str | None = None,
    job_id: str | None = None,
) -> UploadIngestResponse:
    if job_id:
        _set_ingestion_job(job_id, status="parsing", progress=20)
    manifest_path = _safe_api_path(SETTINGS.manifest_path)
    persist_dir = _safe_api_path(SETTINGS.vector_db_path)
    manifest = load_manifest(manifest_path)
    decision = plan_document_ingestion(saved_path, manifest)
    if job_id:
        _set_ingestion_job(job_id, document_id=decision.document_id, document_version=decision.document_version)
    chunks = _chunks_for_path(saved_path, decision.document_version)
    if job_id:
        _set_ingestion_job(job_id, status="chunking", progress=50, chunks_created=len(chunks))
    if safe_workspace_id:
        for chunk in chunks:
            chunk.metadata["workspace_id"] = safe_workspace_id
    for chunk in chunks:
        chunk.metadata["access_roles"] = safe_access_roles
    if job_id:
        _set_ingestion_job(job_id, status="embedding", progress=75)
    vectorstore = build_vector_db(chunks, persist_directory=persist_dir, settings=SETTINGS)
    vector_records = count_records(vectorstore)
    record_document_ingestion(manifest, decision, saved_path, chunk_count=len(chunks))
    manifest["documents"][decision.document_id]["filename"] = safe_name
    manifest["documents"][decision.document_id]["access_roles"] = safe_access_roles
    manifest["documents"][decision.document_id]["owner"] = owner or "unknown"
    if safe_workspace_id:
        manifest["documents"][decision.document_id]["workspace_id"] = safe_workspace_id
    save_manifest(manifest, manifest_path)
    _clear_query_cache()
    response = UploadIngestResponse(
        filename=safe_name,
        saved_path=str(saved_path),
        document_id=decision.document_id,
        document_version=decision.document_version,
        workspace_id=safe_workspace_id or "default",
        access_roles=safe_access_roles,
        status="indexed",
        job_id=job_id,
        progress=100 if job_id else None,
        chunks_created=len(chunks),
        vector_records=vector_records,
        chunk_summary=chunk_token_summary(chunks),
    )
    if job_id:
        _set_ingestion_job(
            job_id,
            status="indexed",
            progress=100,
            workspace_id=safe_workspace_id or "default",
            chunks_created=len(chunks),
            vector_records=vector_records,
        )
    MANAGED_OBSERVABILITY.export_log(
        {
            "event": "ingestion_completed",
            "document_id": decision.document_id,
            "workspace_id": safe_workspace_id or "default",
            "chunks_created": len(chunks),
            "vector_records": vector_records,
        }
    )
    return response


def _run_ingestion_job(job_id: str, saved_path: Path, safe_name: str, workspace_id: str | None, access_roles: list[str], owner: str | None = None) -> None:
    try:
        _index_uploaded_file(saved_path, safe_name, workspace_id, access_roles, owner=owner, job_id=job_id)
    except Exception as exc:
        LOGGER.exception("Background ingestion job failed", extra={"job_id": job_id})
        detail = _classify_ingestion_error(exc)
        MANAGED_OBSERVABILITY.export_log({"event": "ingestion_failed", "job_id": job_id, "error": detail["code"]})
        _set_ingestion_job(job_id, status="failed", progress=100, error=detail["message"])


def _retention_days_for_record(record: dict) -> int:
    for key in ("workspace_retention_days", "org_retention_days", "retention_days"):
        value = record.get(key)
        if value not in (None, ""):
            return int(value)
    return SETTINGS.retention_days


def _record_ingested_at(record: dict) -> datetime | None:
    value = record.get("ingested_at")
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None


def _run_scheduled_retention(persist_dir: str | None = None, now: datetime | None = None) -> RetentionRunResponse:
    now = now or datetime.now(UTC)
    manifest_path = _document_manifest_path()
    manifest = load_manifest(manifest_path)
    expired = []
    for record in manifest.get("documents", {}).values():
        ingested_at = _record_ingested_at(record)
        if ingested_at and ingested_at <= now - timedelta(days=_retention_days_for_record(record)):
            expired.append(record)

    if not expired:
        return RetentionRunResponse(status="ok", documents_deleted=0, files_deleted=0, vector_records_deleted=0)

    vectorstore = load_vectorstore(_safe_api_path(persist_dir or SETTINGS.vector_db_path), settings=SETTINGS)
    files_deleted = 0
    vector_records_deleted = 0
    for record in expired:
        document_id = record["document_id"]
        delete_filter = {"document_id": document_id}
        if record.get("workspace_id"):
            delete_filter["workspace_id"] = record["workspace_id"]
        vector_records_deleted += delete_records_by_metadata(vectorstore, delete_filter) or 0
        source_path = _safe_api_path(record["source_path"])
        if source_path.exists():
            source_path.unlink()
            files_deleted += 1
        manifest["documents"].pop(document_id, None)
        append_audit_event(
            {
                "timestamp": now.isoformat(),
                "event": "scheduled_retention_delete",
                "document_id": document_id,
                "workspace_id": record.get("workspace_id", "default"),
                "retention_days": _retention_days_for_record(record),
            },
            _safe_api_path(DEFAULT_AUDIT_LOG),
        )

    save_manifest(manifest, manifest_path)
    _clear_query_cache()
    return RetentionRunResponse(
        status="ok",
        documents_deleted=len(expired),
        files_deleted=files_deleted,
        vector_records_deleted=vector_records_deleted,
    )


def _retention_scheduler_loop() -> None:
    while True:
        time.sleep(max(1, SETTINGS.retention_schedule_seconds))
        try:
            _run_scheduled_retention()
        except Exception:
            LOGGER.exception("Scheduled retention failed")


def _start_retention_scheduler() -> None:
    global RETENTION_THREAD_STARTED
    if RETENTION_THREAD_STARTED or SETTINGS.retention_schedule_seconds <= 0:
        return
    RETENTION_THREAD_STARTED = True
    threading.Thread(target=_retention_scheduler_loop, daemon=True, name="rag-retention").start()


def _purge_logs() -> int:
    logs_dir = _safe_api_path(PROJECT_ROOT / "logs")
    if logs_dir.is_dir():
        shutil.rmtree(logs_dir)
        return 1
    deleted = 0
    for path in [_safe_api_path(DEFAULT_FEEDBACK_LOG), _safe_api_path(DEFAULT_AUDIT_LOG)]:
        if path.is_file():
            path.unlink()
            deleted += 1
    return deleted


def _retrieve_for_request(query: str, request: QueryRequest, vectorstore, auth_context: AuthContext):
    mode = request.retrieval_mode.lower()
    if mode not in SUPPORTED_RETRIEVAL_MODES:
        raise ValueError(f"Unsupported retrieval mode: {request.retrieval_mode}")
    strategy_reason = "explicit retrieval mode"
    if mode == "auto":
        mode, strategy_reason = select_retrieval_strategy(query)
    documents = []
    if mode in {"exact", "hybrid", "sparse", "reranked"}:
        documents = _candidate_documents(vectorstore, query, request.top_k)
    reranker = None
    if mode == "reranked":
        reranker = build_reranker(
            SETTINGS.reranker_provider,
            SETTINGS.reranker_model,
            SETTINGS.reranker_allow_fallback,
        )
    return mode, strategy_reason, retrieve_by_mode(
        query,
        mode,
        vectorstore=vectorstore,
        documents=documents,
        top_k=request.top_k,
        reranker=reranker,
        metadata_filters=_effective_metadata_filters(request),
        user_roles=auth_context.roles,
    )


def _build_query_payload(request: QueryRequest, auth_context: AuthContext, request_id: str) -> dict:
    try:
        safe_query = validate_query(request.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    requested_mode = request.retrieval_mode.lower()
    if requested_mode not in SUPPORTED_RETRIEVAL_MODES:
        raise HTTPException(status_code=400, detail=f"Unsupported retrieval mode: {request.retrieval_mode}")
    if not RATE_LIMITER.allow("default"):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    safe_session_id = _safe_session_id(request.session_id)
    safe_workspace_id = _safe_workspace_id(request.workspace_id)
    conversation_turns = CONVERSATION_MEMORY.get(safe_session_id, safe_workspace_id, auth_context.cache_scope())
    retrieval_query = build_contextual_query(safe_query, conversation_turns)
    event = TraceEvent(
        request_id=request_id,
        query=safe_query,
        original_query=safe_query,
        rewritten_query=retrieval_query,
    )
    cache_filters = {
        **_effective_metadata_filters(request),
        "_retrieval_mode": requested_mode,
        "_auth_scope": auth_context.cache_scope(),
        "_session_id": safe_session_id,
    }
    if safe_session_id:
        cache_filters["_conversation_turns"] = len(conversation_turns)
    with OTEL.span(
        "rag.cache",
        {
            "rag.request_id": request_id,
            "rag.retrieval_mode": requested_mode,
            "rag.top_k": request.top_k,
        },
    ):
        cached_payload = QUERY_CACHE.get(retrieval_query, request.top_k, cache_filters)
    if cached_payload:
        with OTEL.span("rag.cache.hit", {"rag.request_id": request_id, "rag.cache_hit": True}):
            pass
        cached_payload = {**cached_payload, "request_id": request_id, "cached": True}
        cached_payload["trace"] = {**cached_payload["trace"], "request_id": request_id}
        return cached_payload

    try:
        with trace_latency(event):
            vectorstore = load_vectorstore(_safe_api_path(request.persist_dir), settings=SETTINGS)
            with OTEL.span(
                "rag.retrieval",
                {
                    "rag.request_id": request_id,
                    "rag.retrieval_mode": requested_mode,
                    "rag.top_k": request.top_k,
                },
            ):
                retrieval_mode, retrieval_reason, chunks = _retrieve_for_request(
                    retrieval_query,
                    request,
                    vectorstore,
                    auth_context,
                )
            if retrieval_mode == "reranked":
                with OTEL.span("rag.reranking", {"rag.request_id": request_id, "rag.returned_chunks": len(chunks)}):
                    pass
            with OTEL.span(
                "rag.generation",
                {
                    "rag.request_id": request_id,
                    "rag.retrieved_chunks": len(chunks),
                },
            ):
                response = generate_answer(safe_query, chunks)
            with OTEL.span(
                "rag.citation_enforcement",
                {
                    "rag.request_id": request_id,
                    "rag.citation_count": len(response["citations"]),
                },
            ):
                event.retrieved_chunk_ids = chunk_ids(chunks)
                event.answer = response["answer"]
                event.citations = citation_ids(response["citations"])
            quality = score_runtime_answer(response["answer"], chunks)
            response["citations"] = productize_citations(response["citations"])
            visible_answer = productize_answer_citations(response["answer"], response["citations"])
            if not quality.passed:
                visible_answer = f"{visible_answer}\n\nQuality warning: {'; '.join(quality.reasons)}"
            event.token_usage = response.get("token_usage", {})
            event.token_usage["estimated_cost"] = estimate_llm_cost(event.token_usage)
            append_usage(
                UsageEvent(
                    request_id=request_id,
                    subject=auth_context.subject,
                    org_id=auth_context.tenant_id,
                    workspace_id=safe_workspace_id or "default",
                    prompt_tokens=int(event.token_usage.get("prompt_tokens", 0)),
                    answer_tokens=int(event.token_usage.get("answer_tokens", 0)),
                    estimated_cost=float(event.token_usage.get("estimated_cost", 0.0)),
                ),
                _safe_api_path(DEFAULT_USAGE_LOG),
            )
        payload = {
            "request_id": request_id,
            "answer": visible_answer,
            "citations": response["citations"],
            "quality": quality.to_dict(),
            "retrieval": {
                "mode": retrieval_mode,
                "requested_mode": requested_mode,
                "strategy_reason": retrieval_reason,
                "top_k": request.top_k,
                "returned_chunks": len(chunks),
                "chunk_ids": [chunk.metadata.get("chunk_id") for chunk in chunks],
                "auth_subject": auth_context.subject,
                "auth_roles": sorted(auth_context.roles),
                "tenant_id": auth_context.tenant_id,
                "query": retrieval_query,
                "original_query": safe_query,
                "rewritten_query": retrieval_query,
                "conversation_turns": len(conversation_turns),
            },
            "trace": event.to_log_dict(),
            "cached": False,
        }
        append_audit_event(
            {
                "request_id": request_id,
                "user": auth_context.subject,
                "query": safe_query,
                "retrieval_ids": payload["retrieval"]["chunk_ids"],
                "answer": visible_answer,
                "citations": citation_ids(response["citations"]),
                "model": SETTINGS.llm_model or SETTINGS.llm_provider,
                "latency_ms": event.latency_ms,
            },
            _safe_api_path(DEFAULT_AUDIT_LOG),
        )
        MANAGED_OBSERVABILITY.export_trace(event.to_log_dict())
        QUERY_CACHE.set(retrieval_query, request.top_k, payload, cache_filters)
        CONVERSATION_MEMORY.append(
            safe_session_id,
            safe_workspace_id,
            auth_context.cache_scope(),
            ConversationTurn(
                user=safe_query,
                assistant=response["answer"],
                citations=citation_ids(response["citations"]),
            ),
        )
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("RAG query failed", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail=_classify_query_error(exc, request_id, event)) from exc


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _answer_stream_chunks(answer: str):
    return re.findall(r"\S+\s*", answer)


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@router.get("/llm/health", response_model=LLMHealthResponse)
def llm_health():
    try:
        result = get_model_provider(SETTINGS).llm().health_check()
    except Exception as exc:
        return LLMHealthResponse(status="error", provider=SETTINGS.llm_provider, error=str(exc))
    return LLMHealthResponse(**result)


@router.get("/demo", include_in_schema=False)
def demo():
    return FileResponse(DEMO_DIR / "index.html")


@router.get("/demo/styles.css", include_in_schema=False)
def demo_styles():
    return FileResponse(DEMO_DIR / "styles.css", media_type="text/css")


@router.get("/demo/app.js", include_in_schema=False)
def demo_app():
    return FileResponse(DEMO_DIR / "app.js", media_type="application/javascript")


@router.get("/demo/state.js", include_in_schema=False)
def demo_state():
    return FileResponse(DEMO_DIR / "state.js", media_type="application/javascript")


@router.get("/demo/fonts/{font_name}", include_in_schema=False)
def demo_font(font_name: str):
    font_path = DEMO_DIR / "fonts" / font_name
    if font_path.suffix == ".woff2":
        return FileResponse(font_path, media_type="font/woff2")
    return FileResponse(font_path, media_type="font/ttf")


@router.get("/admin", include_in_schema=False)
def admin_console():
    return FileResponse(DEMO_DIR / "admin.html")


def _render_legal_markdown(markdown: str) -> str:
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Production RAG Legal</title><body>']
    in_list = False
    for line in markdown.splitlines():
        line = line.strip()
        if not line:
            if in_list:
                parts.append("</ul>")
                in_list = False
            continue
        if line.startswith("# "):
            parts.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            parts.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{html.escape(line[2:])}</li>")
        else:
            parts.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        parts.append("</ul>")
    parts.append("</body></html>")
    return "\n".join(parts)


@router.get("/legal/{page}", include_in_schema=False)
def legal_page(page: str):
    filename = LEGAL_PAGES.get(page)
    if filename is None:
        raise HTTPException(status_code=404, detail="legal page not found")
    path = _safe_api_path(LEGAL_DIR / filename)
    return Response(_render_legal_markdown(path.read_text(encoding="utf-8")), media_type="text/html")


@router.get("/demo/admin.js", include_in_schema=False)
def demo_admin_app():
    return FileResponse(DEMO_DIR / "admin.js", media_type="application/javascript")


@router.get("/sources/open", include_in_schema=False)
def open_source(path: str):
    source_path = _safe_api_path(path)
    if not source_path.exists() or not source_path.is_file():
        raise HTTPException(status_code=404, detail="source file not found")
    return FileResponse(source_path)


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    try:
        pdf_path = _safe_api_path(request.pdf_path)
        persist_dir = _safe_api_path(request.persist_dir)
        manifest_path = _safe_api_path(request.manifest_path)
        manifest = load_manifest(manifest_path)
        decision = plan_document_ingestion(pdf_path, manifest)
        if not decision.should_reindex:
            return IngestResponse(
                document_id=decision.document_id,
                document_version=decision.document_version,
                status="skipped",
                reason=decision.reason,
                chunks_created=0,
                min_tokens=None,
                max_tokens=None,
                vector_records=None,
                chunk_summary=None,
            )

        chunks = chunk_pdf(
            pdf_path,
            chunk_size=SETTINGS.chunk_size,
            chunk_overlap=SETTINGS.chunk_overlap,
            document_version=decision.document_version,
        )
        token_counts = [count_tokens(chunk.page_content) for chunk in chunks]
        vector_records = None
        if request.build_vector_db:
            vectorstore = build_vector_db(chunks, persist_directory=persist_dir, settings=SETTINGS)
            vector_records = count_records(vectorstore)
        record_document_ingestion(manifest, decision, pdf_path, chunk_count=len(chunks))
        save_manifest(manifest, manifest_path)
        return IngestResponse(
            document_id=decision.document_id,
            document_version=decision.document_version,
            status="indexed",
            reason=decision.reason,
            chunks_created=len(chunks),
            min_tokens=min(token_counts) if token_counts else None,
            max_tokens=max(token_counts) if token_counts else None,
            vector_records=vector_records,
            chunk_summary=chunk_token_summary(chunks),
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Document ingestion failed")
        raise HTTPException(status_code=400, detail=_classify_ingestion_error(exc)) from exc


@router.post("/upload", response_model=UploadIngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: str | None = Form(default=None),
    access_roles: str | None = Form(default=None),
    background: bool = Form(default=False),
    auth_context: AuthContext = Depends(_auth_context),
):
    safe_workspace_id = _safe_workspace_id(workspace_id)
    safe_access_roles = _safe_access_roles(access_roles)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=_product_error_detail(
                "unsupported_upload_type",
                "Upload a PDF, DOCX, PPTX, Markdown, HTML, CSV, or text file.",
                retry="Choose a supported document type and upload again.",
                actions=["reupload"],
            ),
        )

    safe_name = sanitize_upload_filename(file.filename or f"upload{suffix}")
    if Path(safe_name).suffix.lower() not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=_product_error_detail(
                "unsupported_upload_type",
                "Upload a PDF, DOCX, PPTX, Markdown, HTML, CSV, or text file.",
                retry="Choose a supported document type and upload again.",
                actions=["reupload"],
            ),
        )
    upload_dir = _safe_api_path(PROJECT_ROOT / "data" / "uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / safe_name
    saved_path = _safe_api_path(saved_path)

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail=_product_error_detail(
                "empty_upload",
                "The uploaded file is empty.",
                retry="Choose a non-empty document and upload again.",
                actions=["reupload"],
            ),
        )
    if len(content) > SETTINGS.upload_max_bytes:
        raise HTTPException(status_code=413, detail=f"uploaded file exceeds {SETTINGS.upload_max_bytes} bytes")
    saved_path.write_bytes(content)
    try:
        run_upload_scan(saved_path, SETTINGS.upload_scan_command)
    except ValueError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=_product_error_detail(
                "upload_scan_failed",
                "The upload did not pass the safety scan.",
                retry="Upload a clean export or contact an admin.",
                actions=["reupload", "contact_admin"],
            ),
        ) from exc

    try:
        if background:
            job_id = new_request_id()
            _set_ingestion_job(
                job_id,
                job_id=job_id,
                status="queued",
                progress=0,
                workspace_id=safe_workspace_id or "default",
                owner=auth_context.subject,
                filename=safe_name,
                chunks_created=0,
            )
            thread = threading.Thread(
                target=_run_ingestion_job,
                args=(job_id, saved_path, safe_name, safe_workspace_id, safe_access_roles, auth_context.subject),
                daemon=True,
            )
            thread.start()
            return UploadIngestResponse(
                filename=safe_name,
                saved_path=str(saved_path),
                document_id=Path(safe_name).stem,
                document_version="pending",
                workspace_id=safe_workspace_id or "default",
                access_roles=safe_access_roles,
                status="queued",
                job_id=job_id,
                progress=0,
                chunks_created=0,
                vector_records=None,
                chunk_summary=None,
            )
        return _index_uploaded_file(saved_path, safe_name, safe_workspace_id, safe_access_roles, owner=auth_context.subject)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Document upload failed")
        raise HTTPException(status_code=400, detail=_classify_ingestion_error(exc)) from exc


@router.get("/ingestion-jobs/{job_id}", response_model=IngestionJobResponse)
def ingestion_job(job_id: str):
    job = _get_ingestion_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ingestion job not found")
    return IngestionJobResponse(**job)


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(workspace_id: str | None = None):
    return DocumentListResponse(documents=[DocumentRecordResponse(**record) for record in _document_records(workspace_id)])


@router.get("/index-status", response_model=IndexReadinessResponse)
def index_status(workspace_id: str | None = None):
    return _index_readiness(workspace_id)


@router.get("/admin/status", response_model=AdminStatusResponse)
def admin_status(auth_context: AuthContext = Depends(_admin_context), workspace_id: str | None = None):
    safe_workspace_id = _safe_workspace_id(workspace_id)
    if safe_workspace_id:
        _require_workspace_admin(auth_context, safe_workspace_id)
    else:
        _require_global_admin(auth_context)
    documents = [DocumentRecordResponse(**record) for record in _document_records(workspace_id)]
    failed_jobs = [document.model_dump() for document in documents if document.status not in {"indexed", "skipped"}]
    try:
        vector_records = count_records(load_vectorstore(_safe_api_path(SETTINGS.vector_db_path), settings=SETTINGS))
    except Exception:
        vector_records = None
    return AdminStatusResponse(
        health={"api": "ok", "subject": auth_context.subject},
        index={"vector_records": vector_records, "document_count": len(documents)},
        documents=documents,
        failed_jobs=failed_jobs,
    )


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: str,
    workspace_id: str | None = None,
    persist_dir: str | None = None,
    auth_context: AuthContext = Depends(_admin_context),
):
    safe_workspace_id = _safe_workspace_id(workspace_id)
    manifest_path = _document_manifest_path()
    manifest = load_manifest(manifest_path)
    record = manifest.get("documents", {}).get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    if safe_workspace_id and record.get("workspace_id") != safe_workspace_id:
        raise HTTPException(status_code=404, detail="document not found")
    _require_workspace_admin(auth_context, record.get("workspace_id") or safe_workspace_id)

    vectorstore = load_vectorstore(_safe_api_path(persist_dir or SETTINGS.vector_db_path), settings=SETTINGS)
    delete_filter = {"document_id": document_id}
    if record.get("workspace_id"):
        delete_filter["workspace_id"] = record["workspace_id"]
    deleted_records = delete_records_by_metadata(vectorstore, delete_filter)

    source_path = _safe_api_path(record["source_path"])
    source_path.unlink(missing_ok=True)
    del manifest["documents"][document_id]
    save_manifest(manifest, manifest_path)
    _clear_query_cache()
    _audit_admin_action("admin_document_delete", auth_context, record.get("workspace_id"), document_id)
    return DeleteDocumentResponse(
        document_id=document_id,
        status="deleted",
        vector_records_deleted=deleted_records,
    )


@router.patch("/documents/{document_id}", response_model=DocumentRecordResponse)
def rename_document(
    document_id: str,
    request: RenameDocumentRequest,
    workspace_id: str | None = None,
    auth_context: AuthContext = Depends(_admin_context),
):
    safe_workspace_id = _safe_workspace_id(workspace_id)
    manifest_path = _document_manifest_path()
    manifest = load_manifest(manifest_path)
    record = manifest.get("documents", {}).get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    if safe_workspace_id and record.get("workspace_id") != safe_workspace_id:
        raise HTTPException(status_code=404, detail="document not found")
    _require_workspace_admin(auth_context, record.get("workspace_id") or safe_workspace_id)
    record["filename"] = sanitize_upload_filename(request.filename)
    save_manifest(manifest, manifest_path)
    _audit_admin_action("admin_document_rename", auth_context, record.get("workspace_id"), document_id)
    return DocumentRecordResponse(**_document_response_record(record))


@router.post("/workspaces/{workspace_id}/purge", response_model=PurgeWorkspaceResponse)
def purge_workspace(
    workspace_id: str,
    persist_dir: str | None = None,
    auth_context: AuthContext = Depends(_admin_context),
):
    safe_workspace_id = _safe_workspace_id(workspace_id)
    if safe_workspace_id is None:
        raise HTTPException(status_code=400, detail="workspace_id is required")
    _require_workspace_admin(auth_context, safe_workspace_id)

    manifest_path = _document_manifest_path()
    manifest = load_manifest(manifest_path)
    records = [
        record
        for record in manifest.get("documents", {}).values()
        if record.get("workspace_id") == safe_workspace_id
    ]

    vectorstore = load_vectorstore(_safe_api_path(persist_dir or SETTINGS.vector_db_path), settings=SETTINGS)
    deleted_records = delete_records_by_metadata(vectorstore, {"workspace_id": safe_workspace_id})
    files_deleted = 0
    for record in records:
        source_path = _safe_api_path(record["source_path"])
        if source_path.exists():
            source_path.unlink()
            files_deleted += 1
        manifest["documents"].pop(record["document_id"], None)

    save_manifest(manifest, manifest_path)
    _clear_query_cache()
    conversations_deleted = CONVERSATION_MEMORY.clear_workspace(safe_workspace_id)
    logs_deleted = _purge_logs() if SETTINGS.retention_purge_logs else 0
    _audit_admin_action("admin_workspace_purge", auth_context, safe_workspace_id, safe_workspace_id)
    return PurgeWorkspaceResponse(
        workspace_id=safe_workspace_id,
        documents_deleted=len(records),
        files_deleted=files_deleted,
        vector_records_deleted=deleted_records,
        conversations_deleted=conversations_deleted,
        logs_deleted=logs_deleted,
    )


@router.post("/retention/run", response_model=RetentionRunResponse)
def run_retention(
    persist_dir: str | None = None,
    auth_context: AuthContext = Depends(_admin_context),
):
    _require_global_admin(auth_context)
    _audit_admin_action("admin_retention_run", auth_context, None, "retention")
    return _run_scheduled_retention(persist_dir)


@router.post("/documents/{document_id}/reindex", response_model=UploadIngestResponse)
def reindex_document(
    document_id: str,
    workspace_id: str | None = None,
    persist_dir: str | None = None,
    auth_context: AuthContext = Depends(_admin_context),
):
    safe_workspace_id = _safe_workspace_id(workspace_id)
    manifest_path = _document_manifest_path()
    manifest = load_manifest(manifest_path)
    record = manifest.get("documents", {}).get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    if safe_workspace_id and record.get("workspace_id") != safe_workspace_id:
        raise HTTPException(status_code=404, detail="document not found")
    _require_workspace_admin(auth_context, record.get("workspace_id") or safe_workspace_id)

    source_path = _safe_api_path(record["source_path"])
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="source file not found")

    current_version = int(str(record["document_version"]).lstrip("v") or "1")
    document_version = f"v{current_version + 1}"
    chunks = _chunks_for_path(source_path, document_version)
    workspace = record.get("workspace_id")
    access_roles = record.get("access_roles") or ["public"]
    if workspace:
        for chunk in chunks:
            chunk.metadata["workspace_id"] = workspace
    for chunk in chunks:
        chunk.metadata["access_roles"] = access_roles

    vectorstore = build_vector_db(
        chunks,
        persist_directory=_safe_api_path(persist_dir or SETTINGS.vector_db_path),
        settings=SETTINGS,
    )
    vector_records = count_records(vectorstore)
    decision = plan_document_ingestion(source_path, manifest, document_id=document_id)
    record_document_ingestion(manifest, decision, source_path, chunk_count=len(chunks))
    manifest["documents"][document_id]["document_version"] = document_version
    manifest["documents"][document_id]["filename"] = record.get("filename") or source_path.name
    manifest["documents"][document_id]["workspace_id"] = workspace
    manifest["documents"][document_id]["access_roles"] = access_roles
    save_manifest(manifest, manifest_path)
    _clear_query_cache()
    _audit_admin_action("admin_document_reindex", auth_context, workspace, document_id)
    return UploadIngestResponse(
        filename=record.get("filename") or source_path.name,
        saved_path=str(source_path),
        document_id=document_id,
        document_version=document_version,
        workspace_id=workspace or "default",
        access_roles=access_roles,
        status="indexed",
        chunks_created=len(chunks),
        vector_records=vector_records,
        chunk_summary=chunk_token_summary(chunks),
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, auth_context: AuthContext = Depends(_auth_context)):
    request_id = new_request_id()
    return QueryResponse(**_build_query_payload(request, auth_context, request_id))


@router.post("/query/stream")
def query_stream(request: QueryRequest, auth_context: AuthContext = Depends(_auth_context)):
    request_id = new_request_id()

    def stream_events():
        try:
            payload = _build_query_payload(request, auth_context, request_id)
            yield _sse("start", {"request_id": request_id, "cached": payload.get("cached", False)})
            for chunk in _answer_stream_chunks(payload["answer"]):
                yield _sse("token", {"text": chunk})
            yield _sse("complete", payload)
        except HTTPException as exc:
            detail = {"message": exc.detail} if isinstance(exc.detail, str) else exc.detail
            yield _sse("error", {"status_code": exc.status_code, **detail})

    return StreamingResponse(stream_events(), media_type="text/event-stream")


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    event = FeedbackEvent(
        request_id=request.request_id,
        query=request.query,
        answer=request.answer,
        helpful=request.helpful,
        citations=request.citations,
        latency_ms=request.latency_ms,
        note=request.note,
    )
    append_feedback(event, _safe_api_path(request.feedback_path))
    return FeedbackResponse(status="recorded", request_id=request.request_id)


@router.get("/feedback/events", response_model=FeedbackEventsResponse)
def feedback_events(
    format: str = "json",
    auth_context: AuthContext = Depends(_admin_context),
):
    _require_global_admin(auth_context)
    events = [event.__dict__ for event in load_feedback(_safe_api_path(DEFAULT_FEEDBACK_LOG))]
    if format == "csv":
        fields = ["created_at", "request_id", "query", "answer", "helpful", "citations", "latency_ms", "note"]
        rows = [",".join(fields)]
        for event in events:
            rows.append(",".join(str(event.get(field, "")).replace(",", " ") for field in fields))
        return Response("\n".join(rows) + "\n", media_type="text/csv")
    return FeedbackEventsResponse(events=list(reversed(events)))


@router.get("/monitoring", response_model=MonitoringResponse)
def monitoring(feedback_path: str = str(DEFAULT_FEEDBACK_LOG)):
    return MonitoringResponse(metrics=monitoring_metrics(load_feedback(_safe_api_path(feedback_path))))


@router.get("/observability/dashboard", response_model=ObservabilityDashboardResponse)
def observability_dashboard(
    window_minutes: int = 60,
    workspace_id: str | None = None,
    auth_context: AuthContext = Depends(_admin_context),
):
    safe_workspace_id = _safe_workspace_id(workspace_id)
    if safe_workspace_id:
        _require_workspace_admin(auth_context, safe_workspace_id)
    else:
        _require_global_admin(auth_context)
    return _observability_dashboard(window_minutes, workspace_id)


@router.get("/usage", response_model=UsageResponse)
def usage_report(
    workspace_id: str | None = None,
    auth_context: AuthContext = Depends(_admin_context),
):
    safe_workspace_id = _safe_workspace_id(workspace_id)
    if safe_workspace_id:
        _require_workspace_admin(auth_context, safe_workspace_id)
    else:
        _require_global_admin(auth_context)
    return UsageResponse(usage=usage_summary(load_usage(_safe_api_path(DEFAULT_USAGE_LOG)), safe_workspace_id))


@router.get("/audit", response_model=AuditResponse)
def audit_log(
    format: str = "json",
    limit: int = 100,
    auth_context: AuthContext = Depends(_admin_context),
):
    _require_global_admin(auth_context)
    events = load_audit_events(_safe_api_path(DEFAULT_AUDIT_LOG), limit=max(1, min(limit, 1000)))
    if format == "csv":
        return Response(audit_events_csv(events), media_type="text/csv")
    return AuditResponse(events=events)


@router.get("/evaluation", response_model=EvaluationResponse)
def evaluation_report():
    return EvaluationResponse(**build_evaluation_report())


@router.get("/metrics")
def metrics():
    METRICS.record_provider_failures(LLM_PROVIDER_FAILURES)
    return Response(METRICS.to_prometheus(), media_type="text/plain; version=0.0.4")


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _start_retention_scheduler()
        yield

    app = FastAPI(title="Production RAG API", lifespan=lifespan)

    @app.middleware("http")
    async def record_request_metrics(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        start = time.perf_counter()
        status_code = 500
        try:
            with OTEL.span(
                "http.request",
                {
                    "http.method": request.method,
                    "http.route": request.url.path,
                    "rag.request_id": request_id,
                },
            ):
                response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            METRICS.record_request(status_code, latency_ms)
            log_payload = {
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 3),
                "request_id": request_id,
            }
            LOGGER.info(
                structured_request_log(
                    log_payload["method"],
                    log_payload["path"],
                    log_payload["status_code"],
                    log_payload["latency_ms"],
                    log_payload["request_id"],
                )
            )
            MANAGED_OBSERVABILITY.export_log({"event": "api_request", **log_payload})
            MANAGED_OBSERVABILITY.export_metric(
                {
                    "request_count": METRICS.request_count,
                    "status_counts": METRICS.status_counts,
                    "latency_ms_total": round(METRICS.latency_ms_total, 3),
                }
            )
            if "response" in locals():
                response.headers["X-Request-ID"] = request_id

    app.include_router(router)
    return app


app = create_app()
