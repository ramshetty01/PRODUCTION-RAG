from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.rag.chunking import DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_CHUNK_TOKENS, DEFAULT_DB_PATH, EMBEDDING_MODEL


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


@dataclass(frozen=True)
class RuntimeSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    vector_db_path: str = str(DEFAULT_DB_PATH)
    manifest_path: str = "data/processed/ingestion_manifest.json"
    embedding_model: str = EMBEDDING_MODEL
    chunk_size: int = DEFAULT_CHUNK_TOKENS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    top_k: int = 4
    retrieval_mode: str = "semantic"
    log_level: str = "INFO"
    llm_provider: str = "extractive"
    llm_model: str = ""
    llm_api_key: str = ""
    reranker_provider: str = "lexical"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_allow_fallback: bool = True
    api_keys: str = ""
    auth_mode: str = "auto"
    jwt_secret: str = ""
    jwt_issuer: str = ""
    jwt_audience: str = ""


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


def load_settings(dotenv_path: str | Path | None = ".env") -> RuntimeSettings:
    dotenv_values = load_dotenv(dotenv_path) if dotenv_path else {}
    return RuntimeSettings(
        host=_setting_value(dotenv_values, "RAG_HOST", "0.0.0.0"),
        port=_setting_int(dotenv_values, "RAG_PORT", 8000),
        vector_db_path=_setting_value(dotenv_values, "RAG_VECTOR_DB_PATH", str(DEFAULT_DB_PATH)),
        manifest_path=_setting_value(dotenv_values, "RAG_MANIFEST_PATH", "data/processed/ingestion_manifest.json"),
        embedding_model=_setting_value(dotenv_values, "RAG_EMBEDDING_MODEL", EMBEDDING_MODEL),
        chunk_size=_setting_int(dotenv_values, "RAG_CHUNK_SIZE", DEFAULT_CHUNK_TOKENS),
        chunk_overlap=_setting_int(dotenv_values, "RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP_TOKENS),
        top_k=_setting_int(dotenv_values, "RAG_TOP_K", 4),
        retrieval_mode=_setting_value(dotenv_values, "RAG_RETRIEVAL_MODE", "semantic"),
        log_level=_setting_value(dotenv_values, "RAG_LOG_LEVEL", "INFO"),
        llm_provider=_setting_value(dotenv_values, "RAG_LLM_PROVIDER", "extractive"),
        llm_model=_setting_value(dotenv_values, "RAG_LLM_MODEL", ""),
        llm_api_key=_setting_value(dotenv_values, "RAG_LLM_API_KEY", ""),
        reranker_provider=_setting_value(dotenv_values, "RAG_RERANKER_PROVIDER", "lexical"),
        reranker_model=_setting_value(
            dotenv_values,
            "RAG_RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        reranker_allow_fallback=_setting_bool(dotenv_values, "RAG_RERANKER_ALLOW_FALLBACK", True),
        api_keys=_setting_value(dotenv_values, "RAG_API_KEYS", ""),
        auth_mode=_setting_value(dotenv_values, "RAG_AUTH_MODE", "auto"),
        jwt_secret=_setting_value(dotenv_values, "RAG_JWT_SECRET", ""),
        jwt_issuer=_setting_value(dotenv_values, "RAG_JWT_ISSUER", ""),
        jwt_audience=_setting_value(dotenv_values, "RAG_JWT_AUDIENCE", ""),
    )
