from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


@dataclass
class OpenAICompatibleLLMClient(LLMClient):
    api_key: str = field(repr=False)
    model: str
    endpoint: str
    app_title: str = "Production RAG"
    app_url: str = "https://github.com/ramshetty01/PRODUCTION-RAG"
    timeout_seconds: int = 60
    max_tokens: int = 700
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("RAG_LLM_API_KEY is required for the configured LLM provider")
        if not self.model.strip():
            raise ValueError("RAG_LLM_MODEL is required for the configured LLM provider")

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM provider request failed with status {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM provider request failed: {exc.reason}") from exc

        return self._parse_response(body)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _parse_response(body: str) -> str:
        payload = json.loads(body)
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("LLM provider returned no choices")
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError("LLM provider returned an empty message")
        return content.strip()


@dataclass
class OpenRouterLLMClient(OpenAICompatibleLLMClient):
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        headers["HTTP-Referer"] = self.app_url
        headers["X-Title"] = self.app_title
        return headers


@dataclass
class OpenAILLMClient(OpenAICompatibleLLMClient):
    endpoint: str = "https://api.openai.com/v1/chat/completions"


@dataclass
class ExtractiveLLMClient(LLMClient):
    fallback: str = "The answer is not available in the retrieved context."

    def generate(self, prompt: str) -> str:
        match = re.search(r"\[([^\]]+)\]\nsource: .+?\npage: .+?\n(.+?)(?:\n\n|\Z)", prompt, re.S)
        if not match:
            return self.fallback

        citation_id = match.group(1)
        context_text = " ".join(match.group(2).split())
        sentence = context_text.split(". ")[0].strip()
        if sentence and not sentence.endswith("."):
            sentence = f"{sentence}."
        return f"{sentence} [{citation_id}]"
