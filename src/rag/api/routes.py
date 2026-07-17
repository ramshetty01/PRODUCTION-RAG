from __future__ import annotations

import json
import re
import time
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.rag.auth import AuthContext, authenticate_request, parse_api_keys
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
from src.rag.models import get_model_provider
from src.rag.observability import (
    LOGGER,
    MetricsRegistry,
    TraceEvent,
    chunk_ids,
    citation_ids,
    configure_opentelemetry,
    new_request_id,
    structured_request_log,
    trace_latency,
)
from src.rag.performance import build_query_cache, estimate_llm_cost
from src.rag.reranking import build_reranker
from src.rag.retrieval import DEFAULT_TOP_K, load_vectorstore, retrieve_by_mode, select_retrieval_strategy
from src.rag.security import build_rate_limiter, validate_path, validate_query
from src.rag.vector_store import build_vector_db, count_records, delete_records_by_metadata


SETTINGS = load_settings()
SUPPORTED_RETRIEVAL_MODES = {"auto", "semantic", "exact", "hybrid", "sparse", "reranked"}
AUTH_CONTEXTS = parse_api_keys(SETTINGS.api_keys)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = PROJECT_ROOT / "demo"
SUPPORTED_UPLOAD_SUFFIXES = SUPPORTED_DOCUMENT_SUFFIXES


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
    source: str
    source_path: str
    page: int | None
    chunk_index: int | None
    quote: str


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[CitationResponse]
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


class MonitoringResponse(BaseModel):
    metrics: dict


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
    status: str
    chunks_created: int
    vector_records: int | None
    chunk_summary: dict | None = None


class DocumentRecordResponse(BaseModel):
    document_id: str
    document_version: str
    filename: str | None = None
    workspace_id: str | None = None
    status: str
    chunk_count: int
    source_path: str
    ingested_at: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecordResponse]


class DeleteDocumentResponse(BaseModel):
    document_id: str
    status: str
    vector_records_deleted: int | None = None


router = APIRouter()
QUERY_CACHE = build_query_cache(SETTINGS.cache_backend, SETTINGS.redis_url)
RATE_LIMITER = build_rate_limiter(SETTINGS.rate_limit_backend, SETTINGS.redis_url, max_requests=120, window_seconds=60)
METRICS = MetricsRegistry()
OTEL = configure_opentelemetry(SETTINGS)
CONVERSATION_MEMORY = ConversationMemoryStore(max_turns=SETTINGS.conversation_max_turns)


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
    records = list(manifest.get("documents", {}).values())
    if safe_workspace_id:
        records = [record for record in records if record.get("workspace_id") == safe_workspace_id]
    return sorted(records, key=lambda record: record.get("ingested_at") or "", reverse=True)


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
            event.token_usage = response.get("token_usage", {})
            event.token_usage["estimated_cost"] = estimate_llm_cost(event.token_usage)
        payload = {
            "request_id": request_id,
            "answer": response["answer"],
            "citations": response["citations"],
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
        event.error = type(exc).__name__
        raise HTTPException(status_code=400, detail={"message": str(exc), "trace": event.to_log_dict()}) from exc


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


@router.get("/demo/fonts/{font_name}", include_in_schema=False)
def demo_font(font_name: str):
    font_path = DEMO_DIR / "fonts" / font_name
    if font_path.suffix == ".woff2":
        return FileResponse(font_path, media_type="font/woff2")
    return FileResponse(font_path, media_type="font/ttf")


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
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/upload", response_model=UploadIngestResponse)
async def upload_document(file: UploadFile = File(...), workspace_id: str | None = Form(default=None)):
    safe_workspace_id = _safe_workspace_id(workspace_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail="supported upload types: PDF, DOCX, PPTX, Markdown, HTML, CSV, or text")

    safe_name = Path(file.filename or f"upload{suffix}").name
    upload_dir = _safe_api_path(PROJECT_ROOT / "data" / "uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / safe_name

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    saved_path.write_bytes(content)

    try:
        manifest_path = _safe_api_path(SETTINGS.manifest_path)
        persist_dir = _safe_api_path(SETTINGS.vector_db_path)
        manifest = load_manifest(manifest_path)
        decision = plan_document_ingestion(saved_path, manifest)
        chunks = _chunks_for_path(saved_path, decision.document_version)
        if safe_workspace_id:
            for chunk in chunks:
                chunk.metadata["workspace_id"] = safe_workspace_id
        vectorstore = build_vector_db(chunks, persist_directory=persist_dir, settings=SETTINGS)
        vector_records = count_records(vectorstore)
        record_document_ingestion(manifest, decision, saved_path, chunk_count=len(chunks))
        manifest["documents"][decision.document_id]["filename"] = safe_name
        if safe_workspace_id:
            manifest["documents"][decision.document_id]["workspace_id"] = safe_workspace_id
        save_manifest(manifest, manifest_path)
        _clear_query_cache()
        return UploadIngestResponse(
            filename=safe_name,
            saved_path=str(saved_path),
            document_id=decision.document_id,
            document_version=decision.document_version,
            workspace_id=safe_workspace_id or "default",
            status="indexed",
            chunks_created=len(chunks),
            vector_records=vector_records,
            chunk_summary=chunk_token_summary(chunks),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(workspace_id: str | None = None):
    return DocumentListResponse(documents=[DocumentRecordResponse(**record) for record in _document_records(workspace_id)])


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str, workspace_id: str | None = None, persist_dir: str | None = None):
    safe_workspace_id = _safe_workspace_id(workspace_id)
    manifest_path = _document_manifest_path()
    manifest = load_manifest(manifest_path)
    record = manifest.get("documents", {}).get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    if safe_workspace_id and record.get("workspace_id") != safe_workspace_id:
        raise HTTPException(status_code=404, detail="document not found")

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
    return DeleteDocumentResponse(
        document_id=document_id,
        status="deleted",
        vector_records_deleted=deleted_records,
    )


@router.post("/documents/{document_id}/reindex", response_model=UploadIngestResponse)
def reindex_document(document_id: str, workspace_id: str | None = None, persist_dir: str | None = None):
    safe_workspace_id = _safe_workspace_id(workspace_id)
    manifest_path = _document_manifest_path()
    manifest = load_manifest(manifest_path)
    record = manifest.get("documents", {}).get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    if safe_workspace_id and record.get("workspace_id") != safe_workspace_id:
        raise HTTPException(status_code=404, detail="document not found")

    source_path = _safe_api_path(record["source_path"])
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="source file not found")

    current_version = int(str(record["document_version"]).lstrip("v") or "1")
    document_version = f"v{current_version + 1}"
    chunks = _chunks_for_path(source_path, document_version)
    workspace = record.get("workspace_id")
    if workspace:
        for chunk in chunks:
            chunk.metadata["workspace_id"] = workspace

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
    save_manifest(manifest, manifest_path)
    _clear_query_cache()
    return UploadIngestResponse(
        filename=record.get("filename") or source_path.name,
        saved_path=str(source_path),
        document_id=document_id,
        document_version=document_version,
        workspace_id=workspace or "default",
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
            detail = exc.detail if isinstance(exc.detail, str) else exc.detail.get("message", str(exc.detail))
            yield _sse("error", {"status_code": exc.status_code, "message": detail})

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


@router.get("/monitoring", response_model=MonitoringResponse)
def monitoring(feedback_path: str = str(DEFAULT_FEEDBACK_LOG)):
    return MonitoringResponse(metrics=monitoring_metrics(load_feedback(_safe_api_path(feedback_path))))


@router.get("/evaluation", response_model=EvaluationResponse)
def evaluation_report():
    return EvaluationResponse(**build_evaluation_report())


@router.get("/metrics")
def metrics():
    return Response(METRICS.to_prometheus(), media_type="text/plain; version=0.0.4")


def create_app() -> FastAPI:
    app = FastAPI(title="Production RAG API")

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
            LOGGER.info(
                structured_request_log(
                    request.method,
                    request.url.path,
                    status_code,
                    latency_ms,
                    request_id,
                )
            )
            if "response" in locals():
                response.headers["X-Request-ID"] = request_id

    app.include_router(router)
    return app


app = create_app()
