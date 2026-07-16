# Deployment

## Docker Build

```bash
docker build -t production-rag .
```

## Docker Run

```bash
docker run --rm -p 8000:8000 --env-file .env.example production-rag
```

## Docker Compose

Use the production compose manifest when you want local persistence for ChromaDB
and processed ingestion state:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

## Kubernetes

Kubernetes manifests live in `deploy/kubernetes`. They include a ConfigMap,
Secret example, PersistentVolumeClaim, Deployment with `/health` readiness and
liveness probes, and a ClusterIP Service.

```bash
kubectl apply -f deploy/kubernetes/configmap.yaml
kubectl apply -f deploy/kubernetes/pvc.yaml
kubectl apply -f deploy/kubernetes/deployment.yaml
kubectl apply -f deploy/kubernetes/service.yaml
```

Create a production secret from `deploy/kubernetes/secret.example.yaml` before
starting the Deployment.

## Public Demo Deployment

Use `deploy/render.yaml` to publish the API and browser demo as a Render web
service. The blueprint installs Python dependencies, starts the API through
`deploy/render-start.sh`, mounts persistent storage at
`/var/data/production-rag`, and optionally bootstraps the enterprise demo
corpus on first startup.

Expected public URL shape:

```text
https://production-rag-demo.onrender.com
```

Required reviewer endpoints:

```bash
curl https://production-rag-demo.onrender.com/health
curl https://production-rag-demo.onrender.com/demo
curl https://production-rag-demo.onrender.com/metrics
curl -X POST https://production-rag-demo.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What evidence is required before vendor onboarding?","retrieval_mode":"hybrid","top_k":4}'
```

Secrets must be configured in the hosting provider, not committed to git.
`deploy/render.yaml` marks `RAG_API_KEYS`, `RAG_LLM_API_KEY`, and
`RAG_JWT_SECRET` as manually synced environment variables.

After deployment, run the smoke test:

```bash
python scripts/smoke_deploy.py https://production-rag-demo.onrender.com
```

Rollback by selecting the previous successful deploy in Render, or by reverting
the Git commit that triggered the failed deployment and redeploying.

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
