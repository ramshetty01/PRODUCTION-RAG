from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.chunking import DEFAULT_DB_PATH, DEFAULT_PDF_PATH, chunk_pdf, count_tokens
from src.rag.config import load_settings
from src.rag.generation import generate_answer
from src.rag.ingestion import DEFAULT_MANIFEST, load_manifest, plan_document_ingestion, record_document_ingestion, save_manifest
from src.rag.monitoring import DEFAULT_FEEDBACK_LOG, FeedbackEvent, append_feedback, load_feedback, monitoring_metrics
from src.rag.observability import TraceEvent, chunk_ids, citation_ids, new_request_id, trace_latency
from src.rag.performance import QueryCache, estimate_llm_cost
from src.rag.retrieval import DEFAULT_TOP_K, load_vectorstore, retrieve_chunks
from src.rag.security import RateLimiter, validate_query
from src.rag.vector_store import build_chroma_db, count_records


SETTINGS = load_settings()


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


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=SETTINGS.top_k, gt=0)
    persist_dir: str = Field(default=SETTINGS.vector_db_path)
    metadata_filters: dict | None = None
    user_roles: list[str] = Field(default_factory=lambda: ["public"])


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


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    try:
        manifest = load_manifest(request.manifest_path)
        decision = plan_document_ingestion(request.pdf_path, manifest)
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
            )

        chunks = chunk_pdf(request.pdf_path, document_version=decision.document_version)
        token_counts = [count_tokens(chunk.page_content) for chunk in chunks]
        vector_records = None
        if request.build_vector_db:
            vectorstore = build_chroma_db(chunks, persist_directory=Path(request.persist_dir))
            vector_records = count_records(vectorstore)
        record_document_ingestion(manifest, decision, request.pdf_path, chunk_count=len(chunks))
        save_manifest(manifest, request.manifest_path)
        return IngestResponse(
            document_id=decision.document_id,
            document_version=decision.document_version,
            status="indexed",
            reason=decision.reason,
            chunks_created=len(chunks),
            min_tokens=min(token_counts) if token_counts else None,
            max_tokens=max(token_counts) if token_counts else None,
            vector_records=vector_records,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    request_id = new_request_id()
    try:
        safe_query = validate_query(request.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not RATE_LIMITER.allow("default"):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    event = TraceEvent(request_id=request_id, query=safe_query)
    cached_payload = QUERY_CACHE.get(safe_query, request.top_k, request.metadata_filters)
    if cached_payload:
        cached_payload = {**cached_payload, "request_id": request_id, "cached": True}
        cached_payload["trace"] = {**cached_payload["trace"], "request_id": request_id}
        return QueryResponse(**cached_payload)

    try:
        with trace_latency(event):
            vectorstore = load_vectorstore(request.persist_dir)
            chunks = retrieve_chunks(
                safe_query,
                vectorstore,
                top_k=request.top_k,
                metadata_filters=request.metadata_filters,
                user_roles=set(request.user_roles),
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
                "top_k": request.top_k,
                "returned_chunks": len(chunks),
                "chunk_ids": [chunk.metadata.get("chunk_id") for chunk in chunks],
            },
            "trace": event.to_log_dict(),
            "cached": False,
        }
        QUERY_CACHE.set(safe_query, request.top_k, payload, request.metadata_filters)
        return QueryResponse(**payload)
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
    append_feedback(event, request.feedback_path)
    return FeedbackResponse(status="recorded", request_id=request.request_id)


@router.get("/monitoring", response_model=MonitoringResponse)
def monitoring(feedback_path: str = str(DEFAULT_FEEDBACK_LOG)):
    return MonitoringResponse(metrics=monitoring_metrics(load_feedback(feedback_path)))


def create_app() -> FastAPI:
    app = FastAPI(title="Production RAG API")
    app.include_router(router)
    return app


app = create_app()
