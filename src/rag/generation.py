from __future__ import annotations

from src.rag.citations import attach_citations, extract_cited_ids, format_context_with_citations
from src.rag.llm.client import ExtractiveLLMClient


def build_rag_prompt(query: str, chunks) -> str:
    context = format_context_with_citations(chunks)
    return "\n\n".join(
        [
            "Answer the question using only the retrieved context.",
            "Cite every factual claim with the matching chunk ID in square brackets.",
            f"Question: {query}",
            "Retrieved context:",
            context,
        ]
    )


def generate_answer(query: str, chunks, llm=None) -> dict:
    chunks = list(chunks)
    if not chunks:
        return {
            "answer": "The answer is not available in the retrieved context.",
            "citations": [],
        }

    llm = llm or ExtractiveLLMClient()
    prompt = build_rag_prompt(query, chunks)
    answer = llm.generate(prompt)
    cited_ids = extract_cited_ids(answer)
    return attach_citations(answer, chunks, cited_ids=cited_ids)
