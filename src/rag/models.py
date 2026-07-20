from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from dataclasses import dataclass

from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.config import RuntimeSettings, load_settings
from src.rag.llm.client import (
    ExtractiveLLMClient,
    LLMClient,
    LocalOpenAICompatibleLLMClient,
    LocalSynthesisLLMClient,
    OpenAILLMClient,
    OpenRouterLLMClient,
)


LLM_PROVIDER_FAILURES: dict[str, int] = {}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _record_provider_failure(provider: str) -> None:
    LLM_PROVIDER_FAILURES[provider] = LLM_PROVIDER_FAILURES.get(provider, 0) + 1


@dataclass(frozen=True)
class ModelProvider:
    name: str
    settings: RuntimeSettings

    def embeddings(self):
        raise NotImplementedError

    def llm(self) -> LLMClient:
        raise NotImplementedError


class LocalModelProvider(ModelProvider):
    def embeddings(self):
        if self.settings.embedding_model == "hash":
            return HashEmbeddings()
        return HuggingFaceEmbeddings(model_name=self.settings.embedding_model)

    def llm(self) -> LLMClient:
        return LocalSynthesisLLMClient()


class ExtractiveModelProvider(LocalModelProvider):
    def llm(self) -> LLMClient:
        return ExtractiveLLMClient()


class OpenRouterModelProvider(LocalModelProvider):
    def llm(self) -> LLMClient:
        return OpenRouterLLMClient(
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            timeout_seconds=self.settings.llm_timeout_seconds,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
        )


class OpenAIModelProvider(LocalModelProvider):
    def llm(self) -> LLMClient:
        return OpenAILLMClient(
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            timeout_seconds=self.settings.llm_timeout_seconds,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
        )


class LocalOpenAIModelProvider(LocalModelProvider):
    def llm(self) -> LLMClient:
        return LocalOpenAICompatibleLLMClient(
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            endpoint=self.settings.llm_endpoint or LocalOpenAICompatibleLLMClient.endpoint,
            timeout_seconds=self.settings.llm_timeout_seconds,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
        )


@dataclass
class HashEmbeddings:
    dimensions: int = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[bucket] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


@dataclass
class FallbackLLMClient(LLMClient):
    clients: list[tuple[str, LLMClient]]

    def generate(self, prompt: str) -> str:
        errors = []
        for name, client in self.clients:
            try:
                return client.generate(prompt)
            except Exception as exc:
                _record_provider_failure(name)
                errors.append(f"{name}: {type(exc).__name__}")
        raise RuntimeError(f"All LLM providers failed; degraded mode active ({'; '.join(errors)})")

    def health_check(self) -> dict:
        return {"status": "ok", "provider": "FallbackLLMClient", "providers": [name for name, _ in self.clients]}


PROVIDERS = {
    "local": LocalModelProvider,
    "extractive": ExtractiveModelProvider,
    "local-openai": LocalOpenAIModelProvider,
    "openai": OpenAIModelProvider,
    "openrouter": OpenRouterModelProvider,
}


def _fallback_provider_names(settings: RuntimeSettings) -> list[str]:
    return [name.strip() for name in settings.llm_fallback_providers.split(",") if name.strip()]


def _provider_for_name(name: str, settings: RuntimeSettings) -> ModelProvider:
    provider_class = PROVIDERS.get(name)
    if provider_class is None:
        raise ValueError(f"Unsupported model provider: {name}")
    return provider_class(name=name, settings=replace(settings, llm_provider=name))


def get_model_provider(settings: RuntimeSettings | None = None) -> ModelProvider:
    settings = settings or load_settings()
    primary = _provider_for_name(settings.llm_provider, settings)
    fallback_names = _fallback_provider_names(settings)
    if not fallback_names:
        return primary

    class FallbackModelProvider(type(primary)):
        def llm(self) -> LLMClient:
            clients = [(primary.name, primary.llm())]
            clients.extend((name, _provider_for_name(name, settings).llm()) for name in fallback_names)
            return FallbackLLMClient(clients)

    return FallbackModelProvider(name=settings.llm_provider, settings=settings)
