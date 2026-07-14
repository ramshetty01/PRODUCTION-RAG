from src.rag.config import load_settings


def test_load_settings_uses_defaults_without_dotenv(monkeypatch):
    for key in [
        "RAG_TOP_K",
        "RAG_CHUNK_SIZE",
        "RAG_CHUNK_OVERLAP",
        "RAG_LLM_PROVIDER",
        "RAG_RERANKER_PROVIDER",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = load_settings(dotenv_path=None)

    assert settings.top_k == 4
    assert settings.chunk_size == 700
    assert settings.chunk_overlap == 100
    assert settings.llm_provider == "extractive"
    assert settings.reranker_provider == "lexical"


def test_load_settings_reads_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_TOP_K", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "RAG_VECTOR_DB_PATH=/tmp/chroma",
                "RAG_CHUNK_SIZE=600",
                "RAG_CHUNK_OVERLAP=80",
                "RAG_TOP_K=6",
                "RAG_LLM_PROVIDER=openai",
                "RAG_RERANKER_PROVIDER=cross_encoder",
                "RAG_RERANKER_MODEL=cross-encoder/test",
                "RAG_RERANKER_ALLOW_FALLBACK=false",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(dotenv)

    assert settings.vector_db_path == "/tmp/chroma"
    assert settings.chunk_size == 600
    assert settings.chunk_overlap == 80
    assert settings.top_k == 6
    assert settings.llm_provider == "openai"
    assert settings.reranker_provider == "cross_encoder"
    assert settings.reranker_model == "cross-encoder/test"
    assert settings.reranker_allow_fallback is False


def test_env_example_documents_required_runtime_settings():
    env_example = open(".env.example", encoding="utf-8").read()

    assert "RAG_CHUNK_SIZE=" in env_example
    assert "RAG_CHUNK_OVERLAP=" in env_example
    assert "RAG_TOP_K=" in env_example
    assert "RAG_LLM_PROVIDER=" in env_example
    assert "RAG_RERANKER_PROVIDER=" in env_example
