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

Use the production compose manifest when you want a repeatable local stack with
the API, browser demo/static serving, embedded Chroma persistence, logs, and
processed ingestion state:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Open the local operator surfaces:

```bash
curl http://localhost:8000/health
open http://localhost:8080/demo
open http://localhost:8080/admin
```

The compose stack uses named volumes for local enterprise state:

- `rag_chroma`: embedded Chroma vector index.
- `rag_data`: uploaded documents and ingestion manifest.
- `rag_logs`: audit, feedback, and request logs.

Compose loads `.env.example` first and then an optional local `.env`, so keep
secrets in `.env` and commit only reference values.

## Environment Split

Do not share config, data directories, or secrets between staging and
production. Use the checked-in examples as starting points:

```bash
cp deploy/staging.env.example .env.staging
cp deploy/production.env.example .env.production
```

Staging uses `/var/data/production-rag-staging/...`; production uses
`/var/data/production-rag/...`. Keep API keys, JWT secrets, and model keys in
the deployment platform or secret manager, not in committed env files. See
[Secret Management](SECRETS.md) for the required secret checklist and
`RAG_SECRETS_FILE` mounted-secret option.

Run smoke tests against each environment after deploy:

```bash
python scripts/smoke_deploy.py https://staging-rag.example.com
python scripts/smoke_deploy.py https://production-rag.example.com
```

Background ingestion jobs are persisted to `logs/ingestion_jobs.json` in the
local runtime. The API retries a failed background job once, records `attempts`,
and marks unrecoverable jobs as terminal with a user-facing reason. For larger
deployments, point this same boundary at Redis/RQ, Celery, or Arq.

`RAG_ENVIRONMENT=production` refuses the local default vector and manifest
paths, so `.env.example` cannot accidentally target production data.

Run the ingestion worker profile when you want to seed the default corpus into
the same volumes:

```bash
docker compose -f deploy/docker-compose.yml --profile worker run --rm rag-worker
```

Run the optional local LLM service with Ollama:

```bash
docker compose -f deploy/docker-compose.yml --profile local-llm up --build
```

Then configure `.env` with an OpenAI-compatible Ollama endpoint and model, for
example:

```text
RAG_LLM_PROVIDER=openai-compatible
RAG_LLM_ENDPOINT=http://rag-ollama:11434/v1/chat/completions
RAG_LLM_MODEL=llama3.1
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

## Distributed Tracing

OpenTelemetry tracing is optional for local development and disabled by
default. Enable it in production when an OTLP collector or managed tracing
backend is available:

```bash
RAG_OTEL_ENABLED=true
RAG_OTEL_SERVICE_NAME=production-rag
RAG_OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318/v1/traces
```

For managed dashboards that accept JSON events over HTTPS, enable the built-in
export hook. It sends request logs, request metrics, ingestion events, and RAG
trace payloads to the configured endpoint:

```bash
RAG_OBSERVABILITY_EXPORT_ENABLED=true
RAG_OBSERVABILITY_EXPORT_ENDPOINT=https://observability.example.com/ingest
RAG_OBSERVABILITY_EXPORT_API_KEY=...
```

Alert routing should cover ingestion failures, model errors, high request
latency, elevated 4xx/5xx rates, and vector index health.
Use [Production Alerting](ALERTING.md) for Prometheus/Grafana-ready rules for
ingestion failures, LLM errors, high latency, cost spikes, storage usage, and
auth failures.

Install the optional OpenTelemetry packages in the runtime image when tracing is
enabled:

```bash
python -m pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

The API records spans for `http.request`, `rag.cache`, `rag.cache.hit`,
`rag.retrieval`, `rag.reranking`, `rag.generation`, and
`rag.citation_enforcement`. Spans include the request ID and key RAG attributes
such as retrieval mode, Top K, retrieved chunk count, and citation count.

## Runtime Configuration

Environment variables are documented in `.env.example`:

- `RAG_HOST`: bind host for Uvicorn.
- `RAG_ENVIRONMENT`: `local`, `staging`, or `production`; production requires
  explicit non-local storage paths.
- `RAG_PORT`: API port.
- `RAG_VECTOR_DB_PATH`: generated ChromaDB directory.
- `RAG_VECTOR_BACKEND`: `chroma` by default, or `qdrant` for a managed vector
  service.
- `RAG_VECTOR_COLLECTION`: collection name used by the active vector backend.
- `RAG_QDRANT_URL` and `RAG_QDRANT_API_KEY`: Qdrant connection settings when
  `RAG_VECTOR_BACKEND=qdrant`.
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
- `RAG_QUOTA_MAX_DOCUMENTS_PER_WORKSPACE`,
  `RAG_QUOTA_MAX_STORAGE_BYTES_PER_WORKSPACE`,
  `RAG_QUOTA_MAX_REQUESTS_PER_USER`, `RAG_QUOTA_MAX_TOKENS_PER_USER`, and
  `RAG_QUOTA_MAX_CONCURRENT_JOBS_PER_WORKSPACE`: optional billing and usage
  limits. Leave as `0` to disable a limit.
- `RAG_UPLOAD_MAX_BYTES`, `RAG_UPLOAD_MAX_FILES_PER_USER`, and
  `RAG_UPLOAD_SCAN_COMMAND`: upload safety controls. Files are written to
  `data/quarantine`, MIME/extension checked, scanned when a command is
  configured, then moved to `data/uploads` before indexing.
- `RAG_OTEL_ENABLED`, `RAG_OTEL_SERVICE_NAME`, and
  `RAG_OTEL_EXPORTER_OTLP_ENDPOINT`: optional OpenTelemetry tracing controls.
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
