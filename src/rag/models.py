from __future__ import annotations

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


PROVIDERS = {
    "local": LocalModelProvider,
    "extractive": ExtractiveModelProvider,
    "local-openai": LocalOpenAIModelProvider,
    "openai": OpenAIModelProvider,
    "openrouter": OpenRouterModelProvider,
}


def get_model_provider(settings: RuntimeSettings | None = None) -> ModelProvider:
    settings = settings or load_settings()
    provider_class = PROVIDERS.get(settings.llm_provider)
    if provider_class is None:
        raise ValueError(f"Unsupported model provider: {settings.llm_provider}")
    return provider_class(name=settings.llm_provider, settings=settings)
