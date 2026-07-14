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
    assert "curl http://localhost:8000/health" in docs
