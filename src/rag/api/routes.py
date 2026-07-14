from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.chunking import DEFAULT_DB_PATH, DEFAULT_PDF_PATH, chunk_pdf, count_tokens
from src.rag.generation import generate_answer
from src.rag.retrieval import DEFAULT_TOP_K, load_vectorstore, retrieve_chunks
from src.rag.vector_store import build_chroma_db, count_records


class HealthResponse(BaseModel):
    status: str


class IngestRequest(BaseModel):
    pdf_path: str = Field(default=str(DEFAULT_PDF_PATH))
    persist_dir: str = Field(default=str(DEFAULT_DB_PATH))
    build_vector_db: bool = True


class IngestResponse(BaseModel):
    chunks_created: int
    min_tokens: int | None
    max_tokens: int | None
    vector_records: int | None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, gt=0)
    persist_dir: str = Field(default=str(DEFAULT_DB_PATH))
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
    answer: str
    citations: list[CitationResponse]
    retrieval: dict


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    try:
        chunks = chunk_pdf(request.pdf_path)
        token_counts = [count_tokens(chunk.page_content) for chunk in chunks]
        vector_records = None
        if request.build_vector_db:
            vectorstore = build_chroma_db(chunks, persist_directory=Path(request.persist_dir))
            vector_records = count_records(vectorstore)
        return IngestResponse(
            chunks_created=len(chunks),
            min_tokens=min(token_counts) if token_counts else None,
            max_tokens=max(token_counts) if token_counts else None,
            vector_records=vector_records,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        vectorstore = load_vectorstore(request.persist_dir)
        chunks = retrieve_chunks(
            request.query,
            vectorstore,
            top_k=request.top_k,
            metadata_filters=request.metadata_filters,
            user_roles=set(request.user_roles),
        )
        response = generate_answer(request.query, chunks)
        return QueryResponse(
            answer=response["answer"],
            citations=response["citations"],
            retrieval={
                "top_k": request.top_k,
                "returned_chunks": len(chunks),
                "chunk_ids": [chunk.metadata.get("chunk_id") for chunk in chunks],
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def create_app() -> FastAPI:
    app = FastAPI(title="Production RAG API")
    app.include_router(router)
    return app


app = create_app()
