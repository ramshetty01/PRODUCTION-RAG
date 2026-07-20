import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_restore_smoke_script_checks_restored_state(tmp_path):
    manifest = tmp_path / "data" / "processed" / "ingestion_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"documents": {}}', encoding="utf-8")
    vector_db = tmp_path / "chroma_db"
    uploads = tmp_path / "data" / "uploads"
    logs = tmp_path / "logs"
    for path in [vector_db, uploads, logs]:
        path.mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "restore_smoke.py"),
            "--manifest",
            str(manifest),
            "--vector-db",
            str(vector_db),
            "--uploads",
            str(uploads),
            "--logs",
            str(logs),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "manifest: ok" in result.stdout


def test_dockerfile_starts_api_and_defines_healthcheck():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "uvicorn main:app" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/health" in dockerfile
    assert "EXPOSE 8000" in dockerfile


def test_env_example_documents_runtime_settings():
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    staging = (ROOT / "deploy" / "staging.env.example").read_text(encoding="utf-8")
    production = (ROOT / "deploy" / "production.env.example").read_text(encoding="utf-8")

    assert "RAG_ENVIRONMENT=local" in env_example
    assert "RAG_HOST=" in env_example
    assert "RAG_PORT=" in env_example
    assert "RAG_VECTOR_DB_PATH=" in env_example
    assert "RAG_EMBEDDING_MODEL=" in env_example
    assert "RAG_SECRETS_FILE=" in env_example
    assert "RAG_ENVIRONMENT=staging" in staging
    assert "/var/data/production-rag-staging" in staging
    assert "RAG_API_KEYS=" in staging
    assert "RAG_ENVIRONMENT=production" in production
    assert "/var/data/production-rag/chroma_db" in production
    assert "RAG_JWT_SECRET=" in production
    assert "RAG_LLM_API_KEY=" in production


def test_deployment_docs_include_build_run_and_health_commands():
    docs = (ROOT / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")

    assert "docker build -t production-rag ." in docs
    assert "docker run --rm -p 8000:8000" in docs
    assert "docker compose -f deploy/docker-compose.yml up --build" in docs
    assert "docker compose -f deploy/docker-compose.yml --profile worker run --rm rag-worker" in docs
    assert "docker compose -f deploy/docker-compose.yml --profile local-llm up --build" in docs
    assert "open http://localhost:8080/demo" in docs
    assert "RAG_LLM_ENDPOINT=http://rag-ollama:11434/v1/chat/completions" in docs
    assert "optional local `.env`" in docs
    assert "cp deploy/staging.env.example .env.staging" in docs
    assert "cp deploy/production.env.example .env.production" in docs
    assert "[Secret Management](SECRETS.md)" in docs
    assert "RAG_SECRETS_FILE" in docs
    assert "python scripts/smoke_deploy.py https://staging-rag.example.com" in docs
    assert "python scripts/smoke_deploy.py https://production-rag.example.com" in docs
    assert "RAG_ENVIRONMENT=production" in docs
    assert "RAG_AUTH_MODE=api_key" in docs
    assert "RAG_API_KEYS=public-key:public:tenant-a" in docs
    assert "RAG_AUTH_MODE=jwt" in docs
    assert "RAG_JWT_SECRET=<strong-secret>" in docs
    assert "deploy/kubernetes" in docs
    assert "curl http://localhost:8000/health" in docs

def test_docker_compose_defines_persistent_api_service():
    compose = (ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")

    assert "rag-api:" in compose
    assert "rag-ui:" in compose
    assert "rag-worker:" in compose
    assert "rag-ollama:" in compose
    assert "8000:8000" in compose
    assert "8080:8080" in compose
    assert "11434:11434" in compose
    assert "RAG_VECTOR_DB_PATH: /app/chroma_db" in compose
    assert "RAG_ENVIRONMENT: local" in compose
    assert "path: ../.env" in compose
    assert "required: false" in compose
    assert "rag_chroma:/app/chroma_db" in compose
    assert "rag_data:/app/data" in compose
    assert "rag_logs:/app/logs" in compose
    assert "ollama_data:/root/.ollama" in compose
    assert 'profiles: ["worker"]' in compose
    assert 'profiles: ["local-llm"]' in compose
    assert "scripts/ingest_corpus.py" in compose
    assert "healthcheck:" in compose
    assert "http://127.0.0.1:8080/demo" in compose
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
    assert "RAG_ENVIRONMENT: \"production\"" in configmap
    assert "RAG_VECTOR_DB_PATH: \"/app/chroma_db\"" in configmap
    assert "kind: Secret" in secret
    assert "RAG_API_KEYS:" in secret
    assert "kind: PersistentVolumeClaim" in pvc
    assert "storage: 5Gi" in pvc


def test_security_scan_workflow_audits_dependencies_and_container_surface():
    workflow = (ROOT / ".github" / "workflows" / "security-scan.yml").read_text(encoding="utf-8")

    assert "pip-audit==2.7.3" in workflow
    assert "pip-audit -r requirements.txt" in workflow
    assert "pip-audit-report.json" in workflow
    assert "aquasecurity/trivy-action@v0.36.0" in workflow
    assert "scan-type: fs" in workflow
    assert "severity: CRITICAL,HIGH" in workflow
    assert "trivy-report.json" in workflow
    assert "actions/upload-artifact@v4" in workflow


def test_render_blueprint_defines_public_demo_service():
    render = (ROOT / "deploy" / "render.yaml").read_text(encoding="utf-8")
    start = (ROOT / "deploy" / "render-start.sh").read_text(encoding="utf-8")

    assert "type: web" in render
    assert "name: production-rag-demo" in render
    assert "startCommand: bash deploy/render-start.sh" in render
    assert "healthCheckPath: /health" in render
    assert "mountPath: /var/data/production-rag" in render
    assert "RAG_BOOTSTRAP_DEMO_INDEX" in render
    assert "RAG_API_KEYS" in render
    assert "sync: false" in render
    assert "uvicorn main:app" in start
    assert "scripts/ingest_corpus.py" in start
    assert "--manifest \"$RAG_MANIFEST_PATH\"" in start


def test_public_deployment_docs_include_smoke_test_and_rollback():
    docs = (ROOT / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")
    deploy_readme = (ROOT / "deploy" / "README.md").read_text(encoding="utf-8")
    smoke = (ROOT / "scripts" / "smoke_deploy.py").read_text(encoding="utf-8")

    assert "deploy/render.yaml" in docs
    assert "https://production-rag-demo.onrender.com" in docs
    assert "python scripts/smoke_deploy.py" in docs
    assert "Rollback" in docs
    assert "/health" in smoke
    assert "/demo" in smoke
    assert "/metrics" in smoke
    assert "/query" in smoke
    assert "Render Public Demo" in deploy_readme
