from __future__ import annotations

import re
from dataclasses import dataclass

from src.rag.citations import citation_id_for_chunk, extract_cited_ids
from src.rag.generation import REFUSAL_ANSWER


WORD_PATTERN = re.compile(r"\b[a-z0-9][a-z0-9_-]*\b")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "in",
    "is",
    "it",
    "of",
    "or",
    "the",
    "to",
}


@dataclass(frozen=True)
class RuntimeQuality:
    status: str
    passed: bool
    citation_coverage: float
    evidence_support: float
    confidence: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "passed": self.passed,
            "citation_coverage": self.citation_coverage,
            "evidence_support": self.evidence_support,
            "confidence": self.confidence,
            "reasons": self.reasons,
        }


def _terms(text: str) -> set[str]:
    return {word for word in WORD_PATTERN.findall(text.lower()) if word not in STOP_WORDS and len(word) > 2}


def score_runtime_answer(answer: str, chunks, refusal_answer: str = REFUSAL_ANSWER) -> RuntimeQuality:
    chunks = list(chunks)
    retrieved_ids = {citation_id_for_chunk(chunk) for chunk in chunks}
    cited_ids = set(extract_cited_ids(answer))
    valid_citations = cited_ids & retrieved_ids
    reasons: list[str] = []

    if not chunks:
        return RuntimeQuality("refused", True, 1.0, 1.0, 1.0, ["no retrieved evidence"])
    if answer.strip() == refusal_answer:
        return RuntimeQuality("refused", True, 1.0, 1.0, 1.0, ["answer refused"])

    citation_coverage = len(valid_citations) / len(cited_ids) if cited_ids else 0.0
    evidence_terms = _terms(" ".join(chunk.page_content for chunk in chunks if citation_id_for_chunk(chunk) in valid_citations))
    answer_terms = _terms(re.sub(r"\[[^\[\]]+\]", "", answer))
    evidence_support = len(answer_terms & evidence_terms) / len(answer_terms) if answer_terms else 0.0

    if citation_coverage < 1.0:
        reasons.append("answer has missing or invalid citations")
    if evidence_support < 0.35:
        reasons.append("answer terms are weakly supported by cited evidence")

    confidence = round((citation_coverage + evidence_support) / 2, 2)
    passed = not reasons
    return RuntimeQuality(
        "passed" if passed else "warning",
        passed,
        round(citation_coverage, 2),
        round(evidence_support, 2),
        confidence,
        reasons or ["answer passed runtime quality checks"],
    )
