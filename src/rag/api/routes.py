from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.rag.auth import AuthContext, authenticate_request, parse_api_keys
from src.rag.chunking import DEFAULT_DB_PATH, DEFAULT_PDF_PATH, chunk_pdf, chunk_token_summary, count_tokens
from src.rag.config import load_settings
from src.rag.generation import generate_answer
from src.rag.ingestion import DEFAULT_MANIFEST, load_manifest, plan_document_ingestion, record_document_ingestion, save_manifest
from src.rag.monitoring import DEFAULT_FEEDBACK_LOG, FeedbackEvent, append_feedback, load_feedback, monitoring_metrics
from src.rag.observability import TraceEvent, chunk_ids, citation_ids, new_request_id, trace_latency
from src.rag.performance import QueryCache, estimate_llm_cost
from src.rag.reranking import build_reranker
from src.rag.retrieval import DEFAULT_TOP_K, load_vectorstore, retrieve_by_mode
from src.rag.security import RateLimiter, validate_path, validate_query
from src.rag.vector_store import build_chroma_db, count_records


SETTINGS = load_settings()
SUPPORTED_RETRIEVAL_MODES = {"semantic", "exact", "hybrid", "sparse", "reranked"}
AUTH_CONTEXTS = parse_api_keys(SETTINGS.api_keys)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


router = APIRouter()
QUERY_CACHE = QueryCache()
RATE_LIMITER = RateLimiter(max_requests=120, window_seconds=60)


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


def _retrieve_for_request(query: str, request: QueryRequest, vectorstore, auth_context: AuthContext):
    mode = request.retrieval_mode.lower()
    if mode not in SUPPORTED_RETRIEVAL_MODES:
        raise ValueError(f"Unsupported retrieval mode: {request.retrieval_mode}")
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
    return mode, retrieve_by_mode(
        query,
        mode,
        vectorstore=vectorstore,
        documents=documents,
        top_k=request.top_k,
        reranker=reranker,
        metadata_filters=request.metadata_filters,
        user_roles=auth_context.roles,
    )


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


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
            vectorstore = build_chroma_db(chunks, persist_directory=persist_dir)
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


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, auth_context: AuthContext = Depends(_auth_context)):
    request_id = new_request_id()
    try:
        safe_query = validate_query(request.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    requested_mode = request.retrieval_mode.lower()
    if requested_mode not in SUPPORTED_RETRIEVAL_MODES:
        raise HTTPException(status_code=400, detail=f"Unsupported retrieval mode: {request.retrieval_mode}")
    if not RATE_LIMITER.allow("default"):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    event = TraceEvent(request_id=request_id, query=safe_query)
    cache_filters = {
        **(request.metadata_filters or {}),
        "_retrieval_mode": requested_mode,
        "_auth_scope": auth_context.cache_scope(),
    }
    cached_payload = QUERY_CACHE.get(safe_query, request.top_k, cache_filters)
    if cached_payload:
        cached_payload = {**cached_payload, "request_id": request_id, "cached": True}
        cached_payload["trace"] = {**cached_payload["trace"], "request_id": request_id}
        return QueryResponse(**cached_payload)

    try:
        with trace_latency(event):
            vectorstore = load_vectorstore(_safe_api_path(request.persist_dir))
            retrieval_mode, chunks = _retrieve_for_request(
                safe_query,
                request,
                vectorstore,
                auth_context,
            )
            response = generate_answer(safe_query, chunks)
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
                "top_k": request.top_k,
                "returned_chunks": len(chunks),
                "chunk_ids": [chunk.metadata.get("chunk_id") for chunk in chunks],
                "auth_subject": auth_context.subject,
            },
            "trace": event.to_log_dict(),
            "cached": False,
        }
        QUERY_CACHE.set(safe_query, request.top_k, payload, cache_filters)
        return QueryResponse(**payload)
    except HTTPException:
        raise
    except Exception as exc:
        event.error = type(exc).__name__
        raise HTTPException(status_code=400, detail={"message": str(exc), "trace": event.to_log_dict()}) from exc


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


def create_app() -> FastAPI:
    app = FastAPI(title="Production RAG API")
    app.include_router(router)
    return app


app = create_app()
