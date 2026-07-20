import pytest

from src.rag.config import load_settings


def test_load_settings_uses_defaults_without_dotenv(monkeypatch):
    for key in [
        "RAG_TOP_K",
        "RAG_ENVIRONMENT",
        "RAG_RETRIEVAL_MODE",
        "RAG_CONVERSATION_MAX_TURNS",
        "RAG_RETENTION_DAYS",
        "RAG_RETENTION_PURGE_LOGS",
        "RAG_RETENTION_SCHEDULE_SECONDS",
        "RAG_UPLOAD_MAX_BYTES",
        "RAG_UPLOAD_SCAN_COMMAND",
        "RAG_VECTOR_BACKEND",
        "RAG_VECTOR_COLLECTION",
        "RAG_QDRANT_URL",
        "RAG_QDRANT_API_KEY",
        "RAG_CHUNK_SIZE",
        "RAG_CHUNK_OVERLAP",
        "RAG_LLM_PROVIDER",
        "RAG_LLM_FALLBACK_PROVIDERS",
        "RAG_LLM_ENDPOINT",
        "RAG_LLM_TIMEOUT_SECONDS",
        "RAG_LLM_MAX_TOKENS",
        "RAG_LLM_TEMPERATURE",
        "RAG_RERANKER_PROVIDER",
        "RAG_API_KEYS",
        "RAG_CACHE_BACKEND",
        "RAG_RATE_LIMIT_BACKEND",
        "RAG_REDIS_URL",
        "RAG_AUTH_MODE",
        "RAG_JWT_SECRET",
        "RAG_JWT_ISSUER",
        "RAG_JWT_AUDIENCE",
        "RAG_SUPABASE_URL",
        "RAG_SUPABASE_JWT_SECRET",
        "RAG_OTEL_ENABLED",
        "RAG_OTEL_SERVICE_NAME",
        "RAG_OTEL_EXPORTER_OTLP_ENDPOINT",
        "RAG_OBSERVABILITY_EXPORT_ENABLED",
        "RAG_OBSERVABILITY_EXPORT_ENDPOINT",
        "RAG_OBSERVABILITY_EXPORT_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = load_settings(dotenv_path=None)

    assert settings.environment == "local"
    assert settings.top_k == 4
    assert settings.retrieval_mode == "reranked"
    assert settings.conversation_max_turns == 6
    assert settings.retention_days == 30
    assert settings.retention_purge_logs is True
    assert settings.retention_schedule_seconds == 86400
    assert settings.upload_max_bytes == 10 * 1024 * 1024
    assert settings.upload_scan_command == ""
    assert settings.vector_backend == "chroma"
    assert settings.vector_collection == "rag_chunks"
    assert settings.qdrant_url == ""
    assert settings.qdrant_api_key == ""
    assert settings.chunk_size == 700
    assert settings.chunk_overlap == 100
    assert settings.llm_provider == "extractive"
    assert settings.llm_fallback_providers == ""
    assert settings.llm_endpoint == ""
    assert settings.llm_timeout_seconds == 60
    assert settings.llm_max_tokens == 700
    assert settings.llm_temperature == 0.1
    assert settings.reranker_provider == "lexical"
    assert settings.api_keys == ""
    assert settings.cache_backend == "memory"
    assert settings.rate_limit_backend == "memory"
    assert settings.redis_url == ""
    assert settings.auth_mode == "auto"
    assert settings.jwt_secret == ""
    assert settings.supabase_url == ""
    assert settings.supabase_jwt_secret == ""
    assert settings.otel_enabled is False
    assert settings.otel_service_name == "production-rag"
    assert settings.otel_exporter_otlp_endpoint == ""
    assert settings.observability_export_enabled is False
    assert settings.observability_export_endpoint == ""
    assert settings.observability_export_api_key == ""


def test_load_settings_reads_dotenv_file(tmp_path, monkeypatch):
    for key in [
        "RAG_TOP_K",
        "RAG_RETRIEVAL_MODE",
        "RAG_LLM_PROVIDER",
        "RAG_LLM_ENDPOINT",
        "RAG_LLM_TIMEOUT_SECONDS",
        "RAG_LLM_MAX_TOKENS",
        "RAG_LLM_TEMPERATURE",
    ]:
        monkeypatch.delenv(key, raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "RAG_VECTOR_DB_PATH=/tmp/chroma",
                "RAG_ENVIRONMENT=staging",
                "RAG_SECRETS_FILE=",
                "RAG_VECTOR_BACKEND=qdrant",
                "RAG_VECTOR_COLLECTION=prod_chunks",
                "RAG_QDRANT_URL=https://qdrant.example",
                "RAG_QDRANT_API_KEY=qdrant-key",
                "RAG_METADATA_BACKEND=postgres",
                "RAG_DATABASE_URL=postgresql://localhost/rag",
                "RAG_CHUNK_SIZE=600",
                "RAG_CHUNK_OVERLAP=80",
                "RAG_TOP_K=6",
                "RAG_RETRIEVAL_MODE=hybrid",
                "RAG_CONVERSATION_MAX_TURNS=4",
                "RAG_RETENTION_DAYS=7",
                "RAG_RETENTION_PURGE_LOGS=false",
                "RAG_RETENTION_SCHEDULE_SECONDS=3600",
                "RAG_UPLOAD_MAX_BYTES=1024",
                "RAG_UPLOAD_SCAN_COMMAND=/bin/true",
                "RAG_UPLOAD_MAX_FILES_PER_USER=2",
                "RAG_OBJECT_STORAGE_BACKEND=s3",
                "RAG_OBJECT_STORAGE_BUCKET=rag-documents",
                "RAG_OBJECT_STORAGE_PREFIX=tenant-docs",
                "RAG_OBJECT_STORAGE_ENDPOINT=https://s3.example",
                "RAG_OBJECT_STORAGE_REGION=us-east-1",
                "RAG_QUOTA_MAX_DOCUMENTS_PER_WORKSPACE=2",
                "RAG_QUOTA_MAX_STORAGE_BYTES_PER_WORKSPACE=2048",
                "RAG_QUOTA_MAX_REQUESTS_PER_USER=3",
                "RAG_QUOTA_MAX_TOKENS_PER_USER=1000",
                "RAG_QUOTA_MAX_CONCURRENT_JOBS_PER_WORKSPACE=1",
                "RAG_LLM_PROVIDER=openai",
                "RAG_LLM_FALLBACK_PROVIDERS=local-openai,extractive",
                "RAG_LLM_ENDPOINT=http://localhost:11434/v1/chat/completions",
                "RAG_LLM_TIMEOUT_SECONDS=5",
                "RAG_LLM_MAX_TOKENS=256",
                "RAG_LLM_TEMPERATURE=0.0",
                "RAG_RERANKER_PROVIDER=cross_encoder",
                "RAG_RERANKER_MODEL=cross-encoder/test",
                "RAG_RERANKER_ALLOW_FALLBACK=false",
                "RAG_AUTH_MODE=jwt",
                "RAG_API_KEYS=public-key:public,admin-key:public|admin:tenant-a",
                "RAG_CACHE_BACKEND=redis",
                "RAG_RATE_LIMIT_BACKEND=redis",
                "RAG_REDIS_URL=redis://localhost:6379/0",
                "RAG_JWT_SECRET=secret",
                "RAG_JWT_ISSUER=issuer",
                "RAG_JWT_AUDIENCE=rag-api",
                "RAG_SUPABASE_URL=https://example.supabase.co",
                "RAG_SUPABASE_JWT_SECRET=supabase-secret",
                "RAG_OTEL_ENABLED=true",
                "RAG_OTEL_SERVICE_NAME=rag-test",
                "RAG_OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318/v1/traces",
                "RAG_OBSERVABILITY_EXPORT_ENABLED=true",
                "RAG_OBSERVABILITY_EXPORT_ENDPOINT=https://observability.example.com/ingest",
                "RAG_OBSERVABILITY_EXPORT_API_KEY=obs-key",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(dotenv)

    assert settings.vector_db_path == "/tmp/chroma"
    assert settings.environment == "staging"
    assert settings.secrets_file == ""
    assert settings.vector_backend == "qdrant"
    assert settings.vector_collection == "prod_chunks"
    assert settings.qdrant_url == "https://qdrant.example"
    assert settings.qdrant_api_key == "qdrant-key"
    assert settings.metadata_backend == "postgres"
    assert settings.database_url == "postgresql://localhost/rag"
    assert settings.chunk_size == 600
    assert settings.chunk_overlap == 80
    assert settings.top_k == 6
    assert settings.retrieval_mode == "hybrid"
    assert settings.conversation_max_turns == 4
    assert settings.retention_days == 7
    assert settings.retention_purge_logs is False
    assert settings.retention_schedule_seconds == 3600
    assert settings.upload_max_bytes == 1024
    assert settings.upload_scan_command == "/bin/true"
    assert settings.upload_max_files_per_user == 2
    assert settings.object_storage_backend == "s3"
    assert settings.object_storage_bucket == "rag-documents"
    assert settings.object_storage_prefix == "tenant-docs"
    assert settings.object_storage_endpoint == "https://s3.example"
    assert settings.object_storage_region == "us-east-1"
    assert settings.quota_max_documents_per_workspace == 2
    assert settings.quota_max_storage_bytes_per_workspace == 2048
    assert settings.quota_max_requests_per_user == 3
    assert settings.quota_max_tokens_per_user == 1000
    assert settings.quota_max_concurrent_jobs_per_workspace == 1
    assert settings.llm_provider == "openai"
    assert settings.llm_fallback_providers == "local-openai,extractive"
    assert settings.llm_endpoint == "http://localhost:11434/v1/chat/completions"
    assert settings.llm_timeout_seconds == 5
    assert settings.llm_max_tokens == 256
    assert settings.llm_temperature == 0.0
    assert settings.reranker_provider == "cross_encoder"
    assert settings.reranker_model == "cross-encoder/test"
    assert settings.reranker_allow_fallback is False
    assert settings.api_keys == "public-key:public,admin-key:public|admin:tenant-a"
    assert settings.cache_backend == "redis"
    assert settings.rate_limit_backend == "redis"
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.auth_mode == "jwt"
    assert settings.jwt_secret == "secret"
    assert settings.jwt_issuer == "issuer"
    assert settings.jwt_audience == "rag-api"
    assert settings.supabase_url == "https://example.supabase.co"
    assert settings.supabase_jwt_secret == "supabase-secret"
    assert settings.otel_enabled is True
    assert settings.otel_service_name == "rag-test"
    assert settings.otel_exporter_otlp_endpoint == "http://collector:4318/v1/traces"
    assert settings.observability_export_enabled is True
    assert settings.observability_export_endpoint == "https://observability.example.com/ingest"
    assert settings.observability_export_api_key == "obs-key"


def test_load_settings_reads_mounted_secrets_file(tmp_path, monkeypatch):
    for key in ["RAG_SECRETS_FILE", "RAG_LLM_API_KEY", "RAG_JWT_SECRET"]:
        monkeypatch.delenv(key, raising=False)
    secrets = tmp_path / "secrets.env"
    secrets.write_text("RAG_LLM_API_KEY=mounted-key\nRAG_JWT_SECRET=mounted-secret\n", encoding="utf-8")
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"RAG_SECRETS_FILE={secrets}\nRAG_LLM_API_KEY=local-key\n", encoding="utf-8")

    settings = load_settings(dotenv)

    assert settings.secrets_file == str(secrets)
    assert settings.llm_api_key == "mounted-key"
    assert settings.jwt_secret == "mounted-secret"


def test_production_environment_rejects_local_default_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_VECTOR_DB_PATH", raising=False)
    monkeypatch.delenv("RAG_MANIFEST_PATH", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("RAG_ENVIRONMENT=production\n", encoding="utf-8")

    with pytest.raises(ValueError, match="production environment requires explicit non-local"):
        load_settings(dotenv)


def test_production_environment_requires_fail_closed_auth(tmp_path, monkeypatch):
    for key in ["RAG_VECTOR_DB_PATH", "RAG_MANIFEST_PATH", "RAG_AUTH_MODE", "RAG_API_KEYS", "RAG_JWT_SECRET"]:
        monkeypatch.delenv(key, raising=False)
    base = "\n".join(
        [
            "RAG_ENVIRONMENT=production",
            "RAG_VECTOR_DB_PATH=/srv/rag/chroma",
            "RAG_MANIFEST_PATH=/srv/rag/manifest.json",
        ]
    )
    dotenv = tmp_path / ".env"

    dotenv.write_text(base + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RAG_AUTH_MODE=api_key, jwt, or supabase"):
        load_settings(dotenv)

    dotenv.write_text(base + "\nRAG_AUTH_MODE=api_key\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RAG_API_KEYS"):
        load_settings(dotenv)

    dotenv.write_text(base + "\nRAG_AUTH_MODE=jwt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RAG_JWT_SECRET"):
        load_settings(dotenv)

    dotenv.write_text(base + "\nRAG_AUTH_MODE=api_key\nRAG_API_KEYS=key:public\n", encoding="utf-8")
    assert load_settings(dotenv).auth_mode == "api_key"

    dotenv.write_text(base + "\nRAG_AUTH_MODE=supabase\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RAG_SUPABASE_JWT_SECRET"):
        load_settings(dotenv)

    dotenv.write_text(base + "\nRAG_AUTH_MODE=supabase\nRAG_SUPABASE_JWT_SECRET=secret\n", encoding="utf-8")
    assert load_settings(dotenv).auth_mode == "supabase"


def test_s3_object_storage_requires_runtime_dependency(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("RAG_OBJECT_STORAGE_BACKEND=s3\nRAG_OBJECT_STORAGE_BUCKET=rag-documents\n", encoding="utf-8")

    assert load_settings(dotenv).object_storage_backend == "s3"

    monkeypatch.setattr("src.rag.config.find_spec", lambda module: None if module == "boto3" else object())
    with pytest.raises(ValueError, match="S3 object storage requires boto3"):
        load_settings(dotenv)


def test_qdrant_vector_backend_requires_runtime_dependency(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("RAG_VECTOR_BACKEND=qdrant\nRAG_QDRANT_URL=https://qdrant.example\n", encoding="utf-8")

    assert load_settings(dotenv).vector_backend == "qdrant"

    monkeypatch.setattr("src.rag.config.find_spec", lambda module: None if module == "langchain_qdrant" else object())
    with pytest.raises(ValueError, match="Qdrant vector backend requires langchain-qdrant"):
        load_settings(dotenv)


def test_postgres_metadata_backend_requires_database_url_and_dependency(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("RAG_METADATA_BACKEND=postgres\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RAG_DATABASE_URL"):
        load_settings(dotenv)

    dotenv.write_text("RAG_METADATA_BACKEND=postgres\nRAG_DATABASE_URL=postgresql://localhost/rag\n", encoding="utf-8")
    assert load_settings(dotenv).metadata_backend == "postgres"

    monkeypatch.setattr("src.rag.config.find_spec", lambda module: None if module == "psycopg" else object())
    with pytest.raises(ValueError, match="Postgres metadata backend requires psycopg"):
        load_settings(dotenv)


def test_env_example_documents_required_runtime_settings():
    env_example = open(".env.example", encoding="utf-8").read()

    assert "RAG_CHUNK_SIZE=" in env_example
    assert "RAG_SECRETS_FILE=" in env_example
    assert "RAG_VECTOR_BACKEND=" in env_example
    assert "RAG_VECTOR_COLLECTION=" in env_example
    assert "RAG_QDRANT_URL=" in env_example
    assert "RAG_QDRANT_API_KEY=" in env_example
    assert "RAG_METADATA_BACKEND=" in env_example
    assert "RAG_DATABASE_URL=" in env_example
    assert "RAG_CHUNK_OVERLAP=" in env_example
    assert "RAG_TOP_K=" in env_example
    assert "RAG_RETRIEVAL_MODE=" in env_example
    assert "RAG_CONVERSATION_MAX_TURNS=" in env_example
    assert "RAG_RETENTION_DAYS=" in env_example
    assert "RAG_RETENTION_PURGE_LOGS=" in env_example
    assert "RAG_UPLOAD_MAX_BYTES=" in env_example
    assert "RAG_UPLOAD_SCAN_COMMAND=" in env_example
    assert "RAG_UPLOAD_MAX_FILES_PER_USER=" in env_example
    assert "RAG_OBJECT_STORAGE_BACKEND=" in env_example
    assert "RAG_OBJECT_STORAGE_BUCKET=" in env_example
    assert "RAG_OBJECT_STORAGE_PREFIX=" in env_example
    assert "RAG_OBJECT_STORAGE_ENDPOINT=" in env_example
    assert "RAG_OBJECT_STORAGE_REGION=" in env_example
    assert "RAG_QUOTA_MAX_DOCUMENTS_PER_WORKSPACE=" in env_example
    assert "RAG_QUOTA_MAX_STORAGE_BYTES_PER_WORKSPACE=" in env_example
    assert "RAG_QUOTA_MAX_REQUESTS_PER_USER=" in env_example
    assert "RAG_QUOTA_MAX_TOKENS_PER_USER=" in env_example
    assert "RAG_QUOTA_MAX_CONCURRENT_JOBS_PER_WORKSPACE=" in env_example
    assert "RAG_LLM_PROVIDER=" in env_example
    assert "RAG_LLM_ENDPOINT=" in env_example
    assert "RAG_LLM_TIMEOUT_SECONDS=" in env_example
    assert "RAG_LLM_MAX_TOKENS=" in env_example
    assert "RAG_LLM_TEMPERATURE=" in env_example
    assert "RAG_RERANKER_PROVIDER=" in env_example
    assert "RAG_AUTH_MODE=" in env_example
    assert "RAG_API_KEYS=" in env_example
    assert "RAG_CACHE_BACKEND=" in env_example
    assert "RAG_RATE_LIMIT_BACKEND=" in env_example
    assert "RAG_REDIS_URL=" in env_example
    assert "RAG_JWT_SECRET=" in env_example
    assert "RAG_OTEL_ENABLED=" in env_example
    assert "RAG_OTEL_SERVICE_NAME=" in env_example
    assert "RAG_OTEL_EXPORTER_OTLP_ENDPOINT=" in env_example
    assert "RAG_OBSERVABILITY_EXPORT_ENABLED=" in env_example
    assert "RAG_OBSERVABILITY_EXPORT_ENDPOINT=" in env_example
    assert "RAG_OBSERVABILITY_EXPORT_API_KEY=" in env_example
