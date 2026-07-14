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
    log_level: str = "INFO"
    llm_provider: str = "extractive"
    llm_model: str = ""
    llm_api_key: str = ""


def load_dotenv(path: str | Path = ".env") -> None:
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_settings(dotenv_path: str | Path | None = ".env") -> RuntimeSettings:
    if dotenv_path:
        load_dotenv(dotenv_path)
    return RuntimeSettings(
        host=os.getenv("RAG_HOST", "0.0.0.0"),
        port=_env_int("RAG_PORT", 8000),
        vector_db_path=os.getenv("RAG_VECTOR_DB_PATH", str(DEFAULT_DB_PATH)),
        manifest_path=os.getenv("RAG_MANIFEST_PATH", "data/processed/ingestion_manifest.json"),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", EMBEDDING_MODEL),
        chunk_size=_env_int("RAG_CHUNK_SIZE", DEFAULT_CHUNK_TOKENS),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP_TOKENS),
        top_k=_env_int("RAG_TOP_K", 4),
        log_level=os.getenv("RAG_LOG_LEVEL", "INFO"),
        llm_provider=os.getenv("RAG_LLM_PROVIDER", "extractive"),
        llm_model=os.getenv("RAG_LLM_MODEL", ""),
        llm_api_key=os.getenv("RAG_LLM_API_KEY", ""),
    )
