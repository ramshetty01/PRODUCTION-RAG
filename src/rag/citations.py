from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote


_CITED_ID_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def extract_cited_ids(answer: str) -> list[str]:
    return _CITED_ID_PATTERN.findall(answer)


def _source_name(source: str) -> str:
    return Path(source).name if source else "unknown-source"


def citation_id_for_chunk(chunk) -> str:
    chunk_id = chunk.metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)

    source = _source_name(str(chunk.metadata.get("source", "")))
    page = chunk.metadata.get("page", "unknown")
    chunk_index = chunk.metadata.get("chunk_index", "unknown")
    return f"{Path(source).stem}:p{page}:c{chunk_index}"


def citation_for_chunk(chunk) -> dict:
    citation_id = citation_id_for_chunk(chunk)
    source = str(chunk.metadata.get("source", ""))
    page = chunk.metadata.get("page")
    source_name = _source_name(source)
    page_label = f"page {page}" if page is not None else "page unknown"
    source_url = f"/sources/open?path={quote(source)}" if source else ""
    if source.lower().endswith(".pdf") and page is not None:
        source_url = f"{source_url}#page={int(page) + 1}"
    return {
        "id": citation_id,
        "label": f"{source_name}, {page_label}",
        "source": source_name,
        "source_path": source,
        "page": page,
        "chunk_index": chunk.metadata.get("chunk_index"),
        "snippet": chunk.page_content,
        "context": chunk.page_content,
        "source_url": source_url,
        "quote": chunk.page_content,
    }


def citations_for_chunks(chunks) -> list[dict]:
    return [citation_for_chunk(chunk) for chunk in chunks]


def format_context_with_citations(chunks) -> str:
    blocks = []
    for chunk in chunks:
        citation = citation_for_chunk(chunk)
        page = citation["page"] if citation["page"] is not None else "unknown"
        blocks.append(
            "\n".join(
                [
                    f"[{citation['id']}]",
                    f"source: {citation['source']}",
                    f"page: {page}",
                    chunk.page_content,
                ]
            )
        )
    return "\n\n".join(blocks)


def select_citations(chunks, cited_ids: list[str] | None = None) -> list[dict]:
    citations_by_id = {citation_id_for_chunk(chunk): citation_for_chunk(chunk) for chunk in chunks}
    if cited_ids is None:
        return list(citations_by_id.values())

    return [citations_by_id[citation_id] for citation_id in cited_ids if citation_id in citations_by_id]


def attach_citations(answer: str, chunks, cited_ids: list[str] | None = None) -> dict:
    return {
        "answer": answer,
        "citations": select_citations(chunks, cited_ids=cited_ids),
    }


def productize_citations(citations: list[dict]) -> list[dict]:
    clean = []
    for citation in citations:
        source = citation.get("source") or _source_name(str(citation.get("source_path", "")))
        source_path = str(citation.get("source_path", ""))
        page = citation.get("page")
        page_label = f"page {page}" if page is not None else "page unknown"
        source_url = ""
        if source_path:
            source_url = f"/sources/open?path={quote(source_path)}"
            if source_path.lower().endswith(".pdf") and page is not None:
                source_url = f"{source_url}#page={int(page) + 1}"
        clean.append(
            {
                **citation,
                "label": citation.get("label") or f"{source}, {page_label}",
                "snippet": citation.get("snippet") or citation.get("quote", ""),
                "context": citation.get("context") or citation.get("quote", ""),
                "source_url": citation.get("source_url") or source_url,
            }
        )
    return clean


def productize_answer_citations(answer: str, citations: list[dict]) -> str:
    clean_answer = answer
    for index, citation in enumerate(citations, start=1):
        clean_answer = clean_answer.replace(f"[{citation['id']}]", f"[{index}]")
    return clean_answer
