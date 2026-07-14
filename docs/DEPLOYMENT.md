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

## Metrics

Prometheus-compatible request metrics are exposed at:

```bash
curl http://localhost:8000/metrics
```

The endpoint includes total requests, request counts by HTTP status code, and
cumulative latency in milliseconds. Scrape this endpoint from your metrics
collector and alert on elevated 4xx/5xx rates or rising latency.

## Request Logs

Every HTTP request emits one structured JSON log event through the `rag.api`
logger. The payload includes the request ID, method, path, status code, and
latency in milliseconds. Pass `X-Request-ID` from upstream gateways to preserve
trace continuity; otherwise the API generates a new request ID and returns it in
the response header.

## Runtime Configuration

Environment variables are documented in `.env.example`:

- `RAG_HOST`: bind host for Uvicorn.
- `RAG_PORT`: API port.
- `RAG_VECTOR_DB_PATH`: generated ChromaDB directory.
- `RAG_MANIFEST_PATH`: ingestion manifest location.
- `RAG_EMBEDDING_MODEL`: default embedding model name.
- `RAG_TOP_K`: default retrieval size.
- `RAG_LOG_LEVEL`: application log level.
- `RAG_API_KEYS`: optional comma-separated API key map in
  `key:role1|role2[:tenant]` format. If unset, local development requests run
  as `public`.
- `RAG_AUTH_MODE`: `dev`, `api_key`, or `jwt`.
- `RAG_JWT_SECRET`, `RAG_JWT_ISSUER`, `RAG_JWT_AUDIENCE`: JWT validation
  settings for signed bearer-token deployments.
- Query cache entries are scoped by authenticated subject, tenant, and derived
  roles so restricted answers are not shared across authorization contexts.
- API request paths are resolved under the project root and traversal outside
  that root is rejected. Prefer server-side configured paths in production.
- `RAG_CACHE_BACKEND` and `RAG_RATE_LIMIT_BACKEND`: `memory` by default, or
  `redis` for shared multi-worker deployments.
- `RAG_REDIS_URL`: Redis connection URL used when either backend is `redis`.

Local development can still use the virtual environment:

```bash
.venv/bin/python -m uvicorn main:app --reload
```
