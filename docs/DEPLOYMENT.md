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
- `RAG_API_KEYS`: optional comma-separated API key map in
  `key:role1|role2[:tenant]` format. If unset, local development requests run
  as `public`.
- Query cache entries are scoped by authenticated subject, tenant, and derived
  roles so restricted answers are not shared across authorization contexts.
- API request paths are resolved under the project root and traversal outside
  that root is rejected. Prefer server-side configured paths in production.

Local development can still use the virtual environment:

```bash
.venv/bin/python -m uvicorn main:app --reload
```
