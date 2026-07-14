from __future__ import annotations

from dataclasses import dataclass

from src.rag.citations import citation_id_for_chunk


@dataclass
class ExactMatch:
    document: object
    score: float
    explanation: str


def _metadata_values(document, metadata_fields: list[str]) -> list[str]:
    values = []
    for field in metadata_fields:
        value = document.metadata.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return values


def exact_search(query: str, documents, metadata_fields: list[str] | None = None, top_k: int = 4) -> list[ExactMatch]:
    metadata_fields = metadata_fields or ["source", "document_id", "chunk_id", "section"]
    needle = query.lower().strip()
    matches = []
    for document in documents:
        text = document.page_content.lower()
        metadata_text = " ".join(_metadata_values(document, metadata_fields)).lower()
        score = 0.0
        reasons = []
        if needle and needle in text:
            score += 1.0
            reasons.append("phrase matched chunk text")
        if needle and needle in metadata_text:
            score += 0.5
            reasons.append("phrase matched metadata")
        if score:
            matches.append(
                ExactMatch(
                    document=document,
                    score=score,
                    explanation=f"{citation_id_for_chunk(document)}: {', '.join(reasons)}",
                )
            )
    matches.sort(key=lambda match: match.score, reverse=True)
    return matches[:top_k]
