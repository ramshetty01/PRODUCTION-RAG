from __future__ import annotations

import os
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path

from src.rag.chunking import DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_CHUNK_TOKENS, DEFAULT_DB_PATH, EMBEDDING_MODEL


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


@dataclass(frozen=True)
class RuntimeSettings:
    environment: str = "local"
    host: str = "0.0.0.0"
    port: int = 8000
    vector_db_path: str = str(DEFAULT_DB_PATH)
    vector_backend: str = "chroma"
    vector_collection: str = "rag_chunks"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    metadata_backend: str = "json"
    database_url: str = ""
    manifest_path: str = "data/processed/ingestion_manifest.json"
    embedding_model: str = EMBEDDING_MODEL
    chunk_size: int = DEFAULT_CHUNK_TOKENS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    top_k: int = 4
    retrieval_mode: str = "reranked"
    log_level: str = "INFO"
    conversation_max_turns: int = 6
    retention_days: int = 30
    retention_purge_logs: bool = True
    retention_schedule_seconds: int = 86400
    upload_max_bytes: int = 10 * 1024 * 1024
    upload_scan_command: str = ""
    upload_max_files_per_user: int = 0
    object_storage_backend: str = "local"
    object_storage_bucket: str = ""
    object_storage_prefix: str = "uploads"
    object_storage_endpoint: str = ""
    object_storage_region: str = ""
    quota_max_documents_per_workspace: int = 0
    quota_max_storage_bytes_per_workspace: int = 0
    quota_max_requests_per_user: int = 0
    quota_max_tokens_per_user: int = 0
    quota_max_concurrent_jobs_per_workspace: int = 0
    llm_provider: str = "extractive"
    llm_fallback_providers: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_endpoint: str = ""
    llm_timeout_seconds: int = 60
    llm_max_tokens: int = 700
    llm_temperature: float = 0.1
    reranker_provider: str = "lexical"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_allow_fallback: bool = True
    api_keys: str = ""
    cache_backend: str = "memory"
    rate_limit_backend: str = "memory"
    redis_url: str = ""
    auth_mode: str = "auto"
    jwt_secret: str = ""
    jwt_issuer: str = ""
    jwt_audience: str = ""
    otel_enabled: bool = False
    otel_service_name: str = "production-rag"
    otel_exporter_otlp_endpoint: str = ""
    observability_export_enabled: bool = False
    observability_export_endpoint: str = ""
    observability_export_api_key: str = ""
    secrets_file: str = ""


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def load_secrets_file(path: str | Path = "") -> dict[str, str]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        import json

        return {str(key): str(value) for key, value in json.loads(path.read_text(encoding="utf-8")).items()}
    return load_dotenv(path)


def _setting_value(dotenv_values: dict[str, str], name: str, default: str) -> str:
    return os.getenv(name) or dotenv_values.get(name) or default


def _setting_int(dotenv_values: dict[str, str], name: str, default: int) -> int:
    value = os.getenv(name) or dotenv_values.get(name)
    return int(value) if value not in (None, "") else default


def _setting_bool(dotenv_values: dict[str, str], name: str, default: bool) -> bool:
    value = os.getenv(name) or dotenv_values.get(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _setting_float(dotenv_values: dict[str, str], name: str, default: float) -> float:
    value = os.getenv(name) or dotenv_values.get(name)
    return float(value) if value not in (None, "") else default


def _require_module(module: str, message: str) -> None:
    if find_spec(module) is None:
        raise ValueError(message)


def load_settings(dotenv_path: str | Path | None = ".env") -> RuntimeSettings:
    dotenv_values = load_dotenv(dotenv_path) if dotenv_path else {}
    secrets_file = os.getenv("RAG_SECRETS_FILE") or dotenv_values.get("RAG_SECRETS_FILE") or ""
    dotenv_values = {**dotenv_values, **load_secrets_file(secrets_file)}
    settings = RuntimeSettings(
        environment=_setting_value(dotenv_values, "RAG_ENVIRONMENT", "local"),
        host=_setting_value(dotenv_values, "RAG_HOST", "0.0.0.0"),
        port=_setting_int(dotenv_values, "RAG_PORT", 8000),
        vector_db_path=_setting_value(dotenv_values, "RAG_VECTOR_DB_PATH", str(DEFAULT_DB_PATH)),
        vector_backend=_setting_value(dotenv_values, "RAG_VECTOR_BACKEND", "chroma"),
        vector_collection=_setting_value(dotenv_values, "RAG_VECTOR_COLLECTION", "rag_chunks"),
        qdrant_url=_setting_value(dotenv_values, "RAG_QDRANT_URL", ""),
        qdrant_api_key=_setting_value(dotenv_values, "RAG_QDRANT_API_KEY", ""),
        metadata_backend=_setting_value(dotenv_values, "RAG_METADATA_BACKEND", "json"),
        database_url=_setting_value(dotenv_values, "RAG_DATABASE_URL", ""),
        manifest_path=_setting_value(dotenv_values, "RAG_MANIFEST_PATH", "data/processed/ingestion_manifest.json"),
        embedding_model=_setting_value(dotenv_values, "RAG_EMBEDDING_MODEL", EMBEDDING_MODEL),
        chunk_size=_setting_int(dotenv_values, "RAG_CHUNK_SIZE", DEFAULT_CHUNK_TOKENS),
        chunk_overlap=_setting_int(dotenv_values, "RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP_TOKENS),
        top_k=_setting_int(dotenv_values, "RAG_TOP_K", 4),
        retrieval_mode=_setting_value(dotenv_values, "RAG_RETRIEVAL_MODE", "reranked"),
        log_level=_setting_value(dotenv_values, "RAG_LOG_LEVEL", "INFO"),
        conversation_max_turns=_setting_int(dotenv_values, "RAG_CONVERSATION_MAX_TURNS", 6),
        retention_days=_setting_int(dotenv_values, "RAG_RETENTION_DAYS", 30),
        retention_purge_logs=_setting_bool(dotenv_values, "RAG_RETENTION_PURGE_LOGS", True),
        retention_schedule_seconds=_setting_int(dotenv_values, "RAG_RETENTION_SCHEDULE_SECONDS", 86400),
        upload_max_bytes=_setting_int(dotenv_values, "RAG_UPLOAD_MAX_BYTES", 10 * 1024 * 1024),
        upload_scan_command=_setting_value(dotenv_values, "RAG_UPLOAD_SCAN_COMMAND", ""),
        upload_max_files_per_user=_setting_int(dotenv_values, "RAG_UPLOAD_MAX_FILES_PER_USER", 0),
        object_storage_backend=_setting_value(dotenv_values, "RAG_OBJECT_STORAGE_BACKEND", "local"),
        object_storage_bucket=_setting_value(dotenv_values, "RAG_OBJECT_STORAGE_BUCKET", ""),
        object_storage_prefix=_setting_value(dotenv_values, "RAG_OBJECT_STORAGE_PREFIX", "uploads"),
        object_storage_endpoint=_setting_value(dotenv_values, "RAG_OBJECT_STORAGE_ENDPOINT", ""),
        object_storage_region=_setting_value(dotenv_values, "RAG_OBJECT_STORAGE_REGION", ""),
        quota_max_documents_per_workspace=_setting_int(dotenv_values, "RAG_QUOTA_MAX_DOCUMENTS_PER_WORKSPACE", 0),
        quota_max_storage_bytes_per_workspace=_setting_int(dotenv_values, "RAG_QUOTA_MAX_STORAGE_BYTES_PER_WORKSPACE", 0),
        quota_max_requests_per_user=_setting_int(dotenv_values, "RAG_QUOTA_MAX_REQUESTS_PER_USER", 0),
        quota_max_tokens_per_user=_setting_int(dotenv_values, "RAG_QUOTA_MAX_TOKENS_PER_USER", 0),
        quota_max_concurrent_jobs_per_workspace=_setting_int(dotenv_values, "RAG_QUOTA_MAX_CONCURRENT_JOBS_PER_WORKSPACE", 0),
        llm_provider=_setting_value(dotenv_values, "RAG_LLM_PROVIDER", "extractive"),
        llm_fallback_providers=_setting_value(dotenv_values, "RAG_LLM_FALLBACK_PROVIDERS", ""),
        llm_model=_setting_value(dotenv_values, "RAG_LLM_MODEL", ""),
        llm_api_key=_setting_value(dotenv_values, "RAG_LLM_API_KEY", ""),
        llm_endpoint=_setting_value(dotenv_values, "RAG_LLM_ENDPOINT", ""),
        llm_timeout_seconds=_setting_int(dotenv_values, "RAG_LLM_TIMEOUT_SECONDS", 60),
        llm_max_tokens=_setting_int(dotenv_values, "RAG_LLM_MAX_TOKENS", 700),
        llm_temperature=_setting_float(dotenv_values, "RAG_LLM_TEMPERATURE", 0.1),
        reranker_provider=_setting_value(dotenv_values, "RAG_RERANKER_PROVIDER", "lexical"),
        reranker_model=_setting_value(
            dotenv_values,
            "RAG_RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        reranker_allow_fallback=_setting_bool(dotenv_values, "RAG_RERANKER_ALLOW_FALLBACK", True),
        api_keys=_setting_value(dotenv_values, "RAG_API_KEYS", ""),
        cache_backend=_setting_value(dotenv_values, "RAG_CACHE_BACKEND", "memory"),
        rate_limit_backend=_setting_value(dotenv_values, "RAG_RATE_LIMIT_BACKEND", "memory"),
        redis_url=_setting_value(dotenv_values, "RAG_REDIS_URL", ""),
        auth_mode=_setting_value(dotenv_values, "RAG_AUTH_MODE", "auto"),
        jwt_secret=_setting_value(dotenv_values, "RAG_JWT_SECRET", ""),
        jwt_issuer=_setting_value(dotenv_values, "RAG_JWT_ISSUER", ""),
        jwt_audience=_setting_value(dotenv_values, "RAG_JWT_AUDIENCE", ""),
        otel_enabled=_setting_bool(dotenv_values, "RAG_OTEL_ENABLED", False),
        otel_service_name=_setting_value(dotenv_values, "RAG_OTEL_SERVICE_NAME", "production-rag"),
        otel_exporter_otlp_endpoint=_setting_value(dotenv_values, "RAG_OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        observability_export_enabled=_setting_bool(dotenv_values, "RAG_OBSERVABILITY_EXPORT_ENABLED", False),
        observability_export_endpoint=_setting_value(dotenv_values, "RAG_OBSERVABILITY_EXPORT_ENDPOINT", ""),
        observability_export_api_key=_setting_value(dotenv_values, "RAG_OBSERVABILITY_EXPORT_API_KEY", ""),
        secrets_file=secrets_file,
    )
    if settings.environment == "production" and (
        settings.vector_db_path in {str(DEFAULT_DB_PATH), "./chroma_db", "chroma_db"}
        or settings.manifest_path in {"data/processed/ingestion_manifest.json", "./data/processed/ingestion_manifest.json"}
    ):
        raise ValueError("production environment requires explicit non-local vector and manifest paths")
    if settings.environment == "production":
        if settings.auth_mode not in {"api_key", "jwt"}:
            raise ValueError("production environment requires RAG_AUTH_MODE=api_key or jwt")
        if settings.auth_mode == "api_key" and not settings.api_keys.strip():
            raise ValueError("production API key auth requires RAG_API_KEYS")
        if settings.auth_mode == "jwt" and not settings.jwt_secret.strip():
            raise ValueError("production JWT auth requires RAG_JWT_SECRET")
    if settings.object_storage_backend.lower() == "s3":
        _require_module("boto3", "S3 object storage requires boto3")
    if settings.vector_backend.lower() == "qdrant":
        _require_module("langchain_qdrant", "Qdrant vector backend requires langchain-qdrant")
    if settings.metadata_backend not in {"json", "postgres"}:
        raise ValueError("RAG_METADATA_BACKEND must be json or postgres")
    if settings.metadata_backend == "postgres":
        if not settings.database_url.strip():
            raise ValueError("RAG_DATABASE_URL is required when RAG_METADATA_BACKEND=postgres")
        _require_module("psycopg", "Postgres metadata backend requires psycopg")
    return settings
