import pytest

from src.rag.config import RuntimeSettings
from src.rag.llm.client import (
    ExtractiveLLMClient,
    LLMClient,
    LocalOpenAICompatibleLLMClient,
    LocalSynthesisLLMClient,
    OpenAILLMClient,
    OpenRouterLLMClient,
)
from src.rag.models import FallbackLLMClient, LLM_PROVIDER_FAILURES, get_model_provider


class FakeEmbeddings:
    def __init__(self, model_name):
        self.model_name = model_name


class FailingLLM(LLMClient):
    def generate(self, prompt: str) -> str:
        raise RuntimeError("provider outage")


class WorkingLLM(LLMClient):
    def generate(self, prompt: str) -> str:
        return "ok"


def test_model_provider_returns_configured_extractive_llm():
    settings = RuntimeSettings(llm_provider="extractive")
    provider = get_model_provider(settings)

    assert provider.name == "extractive"
    assert isinstance(provider.llm(), ExtractiveLLMClient)


def test_model_provider_uses_configured_embedding_model(monkeypatch):
    import src.rag.models as models

    monkeypatch.setattr(models, "HuggingFaceEmbeddings", FakeEmbeddings)
    settings = RuntimeSettings(llm_provider="local", embedding_model="custom-embedding-model")

    embeddings = get_model_provider(settings).embeddings()

    assert embeddings.model_name == "custom-embedding-model"


def test_local_model_provider_returns_synthesis_llm():
    settings = RuntimeSettings(llm_provider="local")
    llm = get_model_provider(settings).llm()

    assert isinstance(llm, LocalSynthesisLLMClient)


def test_model_provider_returns_openrouter_llm_client():
    settings = RuntimeSettings(
        llm_provider="openrouter",
        llm_model="provider/model",
        llm_api_key="test-key",
    )

    llm = get_model_provider(settings).llm()

    assert isinstance(llm, OpenRouterLLMClient)
    assert llm.model == "provider/model"
    assert "test-key" not in repr(llm)


def test_model_provider_returns_openai_compatible_llm_client():
    settings = RuntimeSettings(
        llm_provider="openai",
        llm_model="gpt-test",
        llm_api_key="test-key",
    )

    llm = get_model_provider(settings).llm()

    assert isinstance(llm, OpenAILLMClient)
    assert llm.model == "gpt-test"


def test_model_provider_returns_local_openai_compatible_llm_client():
    settings = RuntimeSettings(
        llm_provider="local-openai",
        llm_model="llama3.1:8b",
        llm_endpoint="http://localhost:11434/v1/chat/completions",
        llm_timeout_seconds=5,
        llm_max_tokens=256,
        llm_temperature=0.0,
    )

    llm = get_model_provider(settings).llm()

    assert isinstance(llm, LocalOpenAICompatibleLLMClient)
    assert llm.model == "llama3.1:8b"
    assert llm.endpoint == "http://localhost:11434/v1/chat/completions"
    assert llm.timeout_seconds == 5
    assert llm.max_tokens == 256
    assert llm.temperature == 0.0


def test_openrouter_provider_requires_key_and_model():
    with pytest.raises(ValueError, match="RAG_LLM_API_KEY"):
        get_model_provider(RuntimeSettings(llm_provider="openrouter", llm_model="provider/model")).llm()

    with pytest.raises(ValueError, match="RAG_LLM_MODEL"):
        get_model_provider(RuntimeSettings(llm_provider="openrouter", llm_api_key="test-key")).llm()


def test_model_provider_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported model provider"):
        get_model_provider(RuntimeSettings(llm_provider="unknown"))


def test_fallback_llm_tries_next_provider_and_tracks_failure():
    LLM_PROVIDER_FAILURES.clear()
    llm = FallbackLLMClient([("primary", FailingLLM()), ("fallback", WorkingLLM())])

    assert llm.generate("prompt") == "ok"
    assert LLM_PROVIDER_FAILURES == {"primary": 1}


def test_fallback_llm_reports_degraded_mode_when_all_providers_fail():
    LLM_PROVIDER_FAILURES.clear()
    llm = FallbackLLMClient([("primary", FailingLLM()), ("fallback", FailingLLM())])

    with pytest.raises(RuntimeError, match="All LLM providers failed; degraded mode active"):
        llm.generate("prompt")

    assert LLM_PROVIDER_FAILURES == {"fallback": 1, "primary": 1}


def test_model_provider_builds_fallback_llm_client():
    settings = RuntimeSettings(llm_provider="extractive", llm_fallback_providers="local")

    llm = get_model_provider(settings).llm()

    assert isinstance(llm, FallbackLLMClient)
    assert [name for name, _ in llm.clients] == ["extractive", "local"]
