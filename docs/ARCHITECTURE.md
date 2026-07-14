# Architecture

## Data Flow

1. Source documents are stored in the repo root or `data/raw/`.
2. `scripts/ingest.py` loads the source document and checks the ingestion
   manifest.
3. `src/rag/chunking.py` splits documents into token-aware chunks with metadata.
4. `src/rag/vector_store.py` stores chunk embeddings in generated `chroma_db/`.
5. `src/rag/retrieval.py` loads ChromaDB and retrieves authorized chunks.
6. `src/rag/hybrid_search.py` combines vector and keyword candidates.
7. `src/rag/reranking.py` reranks candidates before final Top K selection.
8. `src/rag/generation.py` builds a prompt from versioned prompt files and
   generates/refuses an answer.
9. `src/rag/citations.py` attaches citations tied to retrieved chunks.
10. `src/rag/api/routes.py` exposes ingestion, querying, health, feedback, and
    monitoring endpoints.
11. `src/rag/observability.py`, `src/rag/monitoring.py`, and
    `src/rag/performance.py` capture traces, feedback, metrics, cache behavior,
    and cost estimates.
12. `evals/run_ragas.py` scores the golden dataset and enforces quality gates in
    CI.

## Module Map

- `src/rag/chunking.py`: PDF loading, token-aware splitting, chunk metadata.
- `src/rag/ingestion.py`: document IDs, content hashes, versioning, manifest.
- `src/rag/vector_store.py`: ChromaDB creation and loading.
- `src/rag/retrieval.py`: vector retrieval, metadata filters, permissions.
- `src/rag/hybrid_search.py`: BM25 keyword scoring plus vector result merging.
- `src/rag/reranking.py`: candidate reranking interface.
- `src/rag/prompts.py`: versioned prompt loading from `prompts/`.
- `src/rag/models.py`: model provider abstraction for embeddings and LLMs.
- `src/rag/generation.py`: answer generation, refusal, token usage.
- `src/rag/security.py`: query validation, injection detection, PII redaction.
- `src/rag/api/routes.py`: FastAPI service.
- `evals/`: golden dataset and offline evaluation.

## Runtime State

Generated state is intentionally excluded from Git:

- `chroma_db/`: local vector database.
- `data/processed/ingestion_manifest.json`: document lifecycle manifest.
- `logs/feedback.jsonl`: feedback events.

These can be rebuilt or regenerated from source documents, configuration, and
runtime usage.

## Configuration

- `.env.example`: safe runtime environment placeholders.
- `configs/settings.toml`: committed evaluation, latency, and cost settings.
- `prompts/*.md`: version-controlled prompt text.

## Service Interfaces

- `GET /health`: service health.
- `POST /ingest`: document ingestion and optional vector rebuild.
- `POST /query`: retrieval and cited answer generation.
- `POST /feedback`: feedback capture.
- `GET /monitoring`: online monitoring metrics.

## Quality Gates

CI runs:

```bash
python -m pytest
python evals/run_ragas.py --config configs/settings.toml
```

The evaluation gate fails when faithfulness is below the configured threshold.
