from __future__ import annotations

from src.rag.citations import (
    attach_citations,
    citation_id_for_chunk,
    extract_cited_ids,
    format_context_with_citations,
)
from src.rag.llm.client import ExtractiveLLMClient
from src.rag.prompts import PromptBundle, build_prompt_from_bundle, load_prompt_bundle

REFUSAL_ANSWER = "The answer is not available in the retrieved context."


def build_rag_prompt(query: str, chunks, prompts: PromptBundle | None = None) -> str:
    context = format_context_with_citations(chunks)
    prompts = prompts or load_prompt_bundle()
    return build_prompt_from_bundle(query=query, context=context, prompts=prompts)


def _refusal_response(refusal_answer: str = REFUSAL_ANSWER) -> dict:
    return {
        "answer": refusal_answer,
        "citations": [],
    }


def enforce_grounded_answer(answer: str, chunks, refusal_answer: str = REFUSAL_ANSWER) -> dict:
    chunks = list(chunks)
    retrieved_ids = {citation_id_for_chunk(chunk) for chunk in chunks}
    cited_ids = extract_cited_ids(answer)
    valid_cited_ids = [citation_id for citation_id in cited_ids if citation_id in retrieved_ids]

    if not chunks or answer.strip() == refusal_answer:
        return _refusal_response(refusal_answer)
    if not valid_cited_ids:
        return _refusal_response(refusal_answer)

    return attach_citations(answer, chunks, cited_ids=valid_cited_ids)


def generate_answer(query: str, chunks, llm=None, prompts: PromptBundle | None = None) -> dict:
    chunks = list(chunks)
    prompts = prompts or load_prompt_bundle()
    if not chunks:
        return _refusal_response(prompts.refusal)

    llm = llm or ExtractiveLLMClient(fallback=prompts.refusal)
    prompt = build_rag_prompt(query, chunks, prompts=prompts)
    answer = llm.generate(prompt)
    response = enforce_grounded_answer(answer, chunks, refusal_answer=prompts.refusal)
    response["token_usage"] = {
        "prompt_tokens": len(prompt.split()),
        "answer_tokens": len(response["answer"].split()),
    }
    return response
