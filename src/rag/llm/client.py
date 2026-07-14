from __future__ import annotations

import re
from dataclasses import dataclass


class LLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


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
