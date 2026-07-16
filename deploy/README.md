# Production Deployment

## Docker Compose

Start the API with persistent local Chroma and processed-data directories:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Verify the service:

```bash
curl http://localhost:8000/health
```

## Kubernetes

Build and publish an image for your cluster, then update
`deploy/kubernetes/deployment.yaml` with the immutable image tag.

Create a real secret from the example before applying manifests:

```bash
cp deploy/kubernetes/secret.example.yaml /tmp/production-rag-secret.yaml
# edit /tmp/production-rag-secret.yaml with production API keys
kubectl apply -f /tmp/production-rag-secret.yaml
kubectl apply -f deploy/kubernetes/configmap.yaml
kubectl apply -f deploy/kubernetes/pvc.yaml
kubectl apply -f deploy/kubernetes/deployment.yaml
kubectl apply -f deploy/kubernetes/service.yaml
```

The Deployment defines readiness and liveness probes on `/health`, CPU and
memory requests/limits, and a PersistentVolumeClaim for `/app/chroma_db`.

## Render Public Demo

`deploy/render.yaml` defines a public web service for the API and browser demo.
Create the service from the blueprint, set any real secrets in Render, and keep
`RAG_BOOTSTRAP_DEMO_INDEX=true` for a reviewer-ready sample index.

The expected URL shape is:

```text
https://production-rag-demo.onrender.com
```

Verify the deployed service:

```bash
python scripts/smoke_deploy.py https://production-rag-demo.onrender.com
```

The smoke test checks `/health`, `/demo`, `/metrics`, and `/query`. Roll back
through Render's previous successful deploy if any check fails after release.
