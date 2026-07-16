#!/usr/bin/env bash
set -euo pipefail

export RAG_PORT="${PORT:-${RAG_PORT:-10000}}"
export RAG_HOST="${RAG_HOST:-0.0.0.0}"
export RAG_VECTOR_DB_PATH="${RAG_VECTOR_DB_PATH:-/var/data/production-rag/chroma_db}"
export RAG_MANIFEST_PATH="${RAG_MANIFEST_PATH:-/var/data/production-rag/ingestion_manifest.json}"

mkdir -p "$(dirname "$RAG_MANIFEST_PATH")" "$RAG_VECTOR_DB_PATH"

if [[ "${RAG_BOOTSTRAP_DEMO_INDEX:-false}" == "true" && ! -f "${RAG_MANIFEST_PATH}" ]]; then
  python scripts/ingest.py --pdf docs.pdf --persist-dir "$RAG_VECTOR_DB_PATH" --manifest "$RAG_MANIFEST_PATH" --build-vector-db
fi

exec python -m uvicorn main:app --host "$RAG_HOST" --port "$RAG_PORT"
