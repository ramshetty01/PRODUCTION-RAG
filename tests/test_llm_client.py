import json
import urllib.error

import pytest

from src.rag.llm.client import LocalOpenAICompatibleLLMClient, OpenAICompatibleLLMClient, OpenRouterLLMClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def close(self):
        pass


def test_openrouter_client_posts_openai_compatible_chat_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"choices": [{"message": {"content": "Answer. [docs:p0:c0]"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenRouterLLMClient(api_key="test-key", model="provider/model", timeout_seconds=7)

    answer = client.generate("Prompt")

    assert answer == "Answer. [docs:p0:c0]"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["timeout"] == 7
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["headers"]["Http-referer"] == "https://github.com/ramshetty01/PRODUCTION-RAG"
    assert captured["headers"]["X-title"] == "Production RAG"
    assert captured["payload"]["model"] == "provider/model"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "Prompt"}]
    assert captured["payload"]["temperature"] == 0.1


def test_openai_compatible_client_rejects_empty_response(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse({"choices": []}))
    client = OpenAICompatibleLLMClient(
        api_key="test-key",
        model="provider/model",
        endpoint="https://example.test/chat",
    )

    with pytest.raises(RuntimeError, match="returned no choices"):
        client.generate("Prompt")


def test_openai_compatible_client_wraps_http_errors(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=FakeResponse({"error": {"message": "bad key"}}),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleLLMClient(
        api_key="test-key",
        model="provider/model",
        endpoint="https://example.test/chat",
    )

    with pytest.raises(RuntimeError, match="status 401"):
        client.generate("Prompt")


def test_local_openai_compatible_client_allows_missing_api_key(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"choices": [{"message": {"content": "Local answer. [docs:p0:c0]"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = LocalOpenAICompatibleLLMClient(model="llama3.1:8b")

    answer = client.generate("Prompt")

    assert answer == "Local answer. [docs:p0:c0]"
    assert "Authorization" not in captured["headers"]
    assert captured["payload"]["model"] == "llama3.1:8b"


def test_openai_compatible_health_check_uses_models_endpoint(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse({"data": [{"id": "llama"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = LocalOpenAICompatibleLLMClient(
        model="llama",
        endpoint="http://localhost:8001/v1/chat/completions",
        timeout_seconds=3,
    )

    result = client.health_check()

    assert result["status"] == "ok"
    assert result["model"] == "llama"
    assert captured["url"] == "http://localhost:8001/v1/models"
    assert captured["timeout"] == 3
