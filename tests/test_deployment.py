from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_starts_api_and_defines_healthcheck():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "uvicorn main:app" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/health" in dockerfile
    assert "EXPOSE 8000" in dockerfile


def test_env_example_documents_runtime_settings():
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "RAG_HOST=" in env_example
    assert "RAG_PORT=" in env_example
    assert "RAG_VECTOR_DB_PATH=" in env_example
    assert "RAG_EMBEDDING_MODEL=" in env_example


def test_deployment_docs_include_build_run_and_health_commands():
    docs = (ROOT / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")

    assert "docker build -t production-rag ." in docs
    assert "docker run --rm -p 8000:8000" in docs
    assert "docker compose -f deploy/docker-compose.yml up --build" in docs
    assert "deploy/kubernetes" in docs
    assert "curl http://localhost:8000/health" in docs


def test_docker_compose_defines_persistent_api_service():
    compose = (ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")

    assert "rag-api:" in compose
    assert "8000:8000" in compose
    assert "RAG_VECTOR_DB_PATH: /app/chroma_db" in compose
    assert "../chroma_db:/app/chroma_db" in compose
    assert "../data:/app/data" in compose
    assert "healthcheck:" in compose
    assert "restart: unless-stopped" in compose


def test_kubernetes_manifests_define_production_runtime_contract():
    deployment = (ROOT / "deploy" / "kubernetes" / "deployment.yaml").read_text(encoding="utf-8")
    service = (ROOT / "deploy" / "kubernetes" / "service.yaml").read_text(encoding="utf-8")
    configmap = (ROOT / "deploy" / "kubernetes" / "configmap.yaml").read_text(encoding="utf-8")
    secret = (ROOT / "deploy" / "kubernetes" / "secret.example.yaml").read_text(encoding="utf-8")
    pvc = (ROOT / "deploy" / "kubernetes" / "pvc.yaml").read_text(encoding="utf-8")

    assert "kind: Deployment" in deployment
    assert "readinessProbe:" in deployment
    assert "livenessProbe:" in deployment
    assert "resources:" in deployment
    assert "persistentVolumeClaim:" in deployment
    assert "claimName: production-rag-chroma" in deployment
    assert "kind: Service" in service
    assert "type: ClusterIP" in service
    assert "kind: ConfigMap" in configmap
    assert "RAG_VECTOR_DB_PATH: \"/app/chroma_db\"" in configmap
    assert "kind: Secret" in secret
    assert "RAG_API_KEYS:" in secret
    assert "kind: PersistentVolumeClaim" in pvc
    assert "storage: 5Gi" in pvc
