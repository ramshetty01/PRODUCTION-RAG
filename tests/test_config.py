from src.rag.config import load_settings


def test_load_settings_uses_defaults_without_dotenv(monkeypatch):
    for key in [
        "RAG_TOP_K",
        "RAG_RETRIEVAL_MODE",
        "RAG_CONVERSATION_MAX_TURNS",
        "RAG_VECTOR_BACKEND",
        "RAG_VECTOR_COLLECTION",
        "RAG_QDRANT_URL",
        "RAG_QDRANT_API_KEY",
        "RAG_CHUNK_SIZE",
        "RAG_CHUNK_OVERLAP",
        "RAG_LLM_PROVIDER",
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
        "RAG_OTEL_ENABLED",
        "RAG_OTEL_SERVICE_NAME",
        "RAG_OTEL_EXPORTER_OTLP_ENDPOINT",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = load_settings(dotenv_path=None)

    assert settings.top_k == 4
    assert settings.retrieval_mode == "reranked"
    assert settings.conversation_max_turns == 6
    assert settings.vector_backend == "chroma"
    assert settings.vector_collection == "rag_chunks"
    assert settings.qdrant_url == ""
    assert settings.qdrant_api_key == ""
    assert settings.chunk_size == 700
    assert settings.chunk_overlap == 100
    assert settings.llm_provider == "extractive"
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
    assert settings.otel_enabled is False
    assert settings.otel_service_name == "production-rag"
    assert settings.otel_exporter_otlp_endpoint == ""


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
                "RAG_VECTOR_BACKEND=qdrant",
                "RAG_VECTOR_COLLECTION=prod_chunks",
                "RAG_QDRANT_URL=https://qdrant.example",
                "RAG_QDRANT_API_KEY=qdrant-key",
                "RAG_CHUNK_SIZE=600",
                "RAG_CHUNK_OVERLAP=80",
                "RAG_TOP_K=6",
                "RAG_RETRIEVAL_MODE=hybrid",
                "RAG_CONVERSATION_MAX_TURNS=4",
                "RAG_LLM_PROVIDER=openai",
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
                "RAG_OTEL_ENABLED=true",
                "RAG_OTEL_SERVICE_NAME=rag-test",
                "RAG_OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318/v1/traces",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(dotenv)

    assert settings.vector_db_path == "/tmp/chroma"
    assert settings.vector_backend == "qdrant"
    assert settings.vector_collection == "prod_chunks"
    assert settings.qdrant_url == "https://qdrant.example"
    assert settings.qdrant_api_key == "qdrant-key"
    assert settings.chunk_size == 600
    assert settings.chunk_overlap == 80
    assert settings.top_k == 6
    assert settings.retrieval_mode == "hybrid"
    assert settings.conversation_max_turns == 4
    assert settings.llm_provider == "openai"
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
    assert settings.otel_enabled is True
    assert settings.otel_service_name == "rag-test"
    assert settings.otel_exporter_otlp_endpoint == "http://collector:4318/v1/traces"


def test_env_example_documents_required_runtime_settings():
    env_example = open(".env.example", encoding="utf-8").read()

    assert "RAG_CHUNK_SIZE=" in env_example
    assert "RAG_VECTOR_BACKEND=" in env_example
    assert "RAG_VECTOR_COLLECTION=" in env_example
    assert "RAG_QDRANT_URL=" in env_example
    assert "RAG_QDRANT_API_KEY=" in env_example
    assert "RAG_CHUNK_OVERLAP=" in env_example
    assert "RAG_TOP_K=" in env_example
    assert "RAG_RETRIEVAL_MODE=" in env_example
    assert "RAG_CONVERSATION_MAX_TURNS=" in env_example
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
