# Deployment

## Docker Build

```bash
docker build -t production-rag .
```

## Docker Run

```bash
docker run --rm -p 8000:8000 --env-file .env.example production-rag
```

## Health Check

The API exposes:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

The Docker image includes a `HEALTHCHECK` that calls `/health` inside the
container.

## Runtime Configuration

Environment variables are documented in `.env.example`:

- `RAG_HOST`: bind host for Uvicorn.
- `RAG_PORT`: API port.
- `RAG_VECTOR_DB_PATH`: generated ChromaDB directory.
- `RAG_MANIFEST_PATH`: ingestion manifest location.
- `RAG_EMBEDDING_MODEL`: default embedding model name.
- `RAG_TOP_K`: default retrieval size.
- `RAG_LOG_LEVEL`: application log level.

Local development can still use the virtual environment:

```bash
.venv/bin/python -m uvicorn main:app --reload
```
