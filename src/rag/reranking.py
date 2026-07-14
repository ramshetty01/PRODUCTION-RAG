from __future__ import annotations

from dataclasses import dataclass

from src.rag.hybrid_search import tokenize


@dataclass
class RerankResult:
    document: object
    score: float


class LexicalReranker:
    def score(self, query: str, document) -> float:
        query_terms = set(tokenize(query))
        doc_terms = tokenize(document.page_content)
        if not query_terms or not doc_terms:
            return 0.0

        doc_term_set = set(doc_terms)
        overlap = len(query_terms & doc_term_set)
        exact_phrase_bonus = 1.0 if query.lower() in document.page_content.lower() else 0.0
        return overlap / len(query_terms) + exact_phrase_bonus


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, document) -> float:
        return float(self.model.predict([(query, document.page_content)])[0])


def rerank_chunks(query: str, chunks, top_k: int = 4, reranker=None):
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    reranker = reranker or LexicalReranker()
    scored = [RerankResult(document=chunk, score=reranker.score(query, chunk)) for chunk in chunks]
    scored.sort(key=lambda result: result.score, reverse=True)
    return [result.document for result in scored[:top_k]]
