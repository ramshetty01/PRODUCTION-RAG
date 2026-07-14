from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = PROJECT_ROOT / "prompts"


@dataclass(frozen=True)
class PromptBundle:
    system: str
    citation: str
    refusal: str
    version: str = "markdown-v1"


def load_prompt(name: str, prompt_dir: str | Path = PROMPT_DIR) -> str:
    path = Path(prompt_dir) / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_prompt_bundle(prompt_dir: str | Path = PROMPT_DIR) -> PromptBundle:
    return PromptBundle(
        system=load_prompt("system", prompt_dir=prompt_dir),
        citation=load_prompt("citation", prompt_dir=prompt_dir),
        refusal=load_prompt("refusal", prompt_dir=prompt_dir),
    )


def build_prompt_from_bundle(query: str, context: str, prompts: PromptBundle) -> str:
    return "\n\n".join(
        [
            f"Prompt-Version: {prompts.version}",
            prompts.system,
            prompts.citation,
            f"Refusal response: {prompts.refusal}",
            "<user_question>",
            f"Question: {query}",
            "</user_question>",
            "Retrieved context:",
            "<retrieved_context>",
            context,
            "</retrieved_context>",
        ]
    )
