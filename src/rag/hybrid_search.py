from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from src.rag.citations import citation_id_for_chunk


TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class SearchResult:
    document: object
    score: float
    source: str


class BM25Index:
    def __init__(self, documents, k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(doc.page_content)) for doc in self.documents]
        self.doc_lengths = [sum(freqs.values()) for freqs in self.term_frequencies]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.document_frequencies = Counter()
        for freqs in self.term_frequencies:
            self.document_frequencies.update(freqs.keys())

    def score_document(self, query_tokens: list[str], index: int) -> float:
        if not self.documents or not query_tokens:
            return 0.0

        score = 0.0
        freqs = self.term_frequencies[index]
        doc_length = self.doc_lengths[index] or 1
        corpus_size = len(self.documents)

        for token in query_tokens:
            term_frequency = freqs.get(token, 0)
            if term_frequency == 0:
                continue

            doc_frequency = self.document_frequencies[token]
            idf = math.log(1 + (corpus_size - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * doc_length / (self.avg_doc_length or 1)
            )
            score += idf * (term_frequency * (self.k1 + 1)) / denominator
        return score

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        query_tokens = tokenize(query)
        scored = [
            SearchResult(document=doc, score=self.score_document(query_tokens, index), source="bm25")
            for index, doc in enumerate(self.documents)
        ]
        scored = [result for result in scored if result.score > 0]
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]


def _normalize_scores(results: list[SearchResult]) -> dict[str, float]:
    if not results:
        return {}
    max_score = max(result.score for result in results) or 1.0
    return {citation_id_for_chunk(result.document): result.score / max_score for result in results}


def hybrid_search(
    query: str,
    vectorstore,
    keyword_documents,
    top_k: int = 4,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
):
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    vector_docs = vectorstore.similarity_search(query, k=top_k) if vectorstore is not None else []
    vector_results = [
        SearchResult(document=doc, score=float(top_k - index), source="vector")
        for index, doc in enumerate(vector_docs)
    ]
    keyword_results = BM25Index(keyword_documents).search(query, top_k=top_k)

    normalized_vector = _normalize_scores(vector_results)
    normalized_keyword = _normalize_scores(keyword_results)
    docs_by_id = {}
    scores_by_id = {}

    for result in [*vector_results, *keyword_results]:
        chunk_id = citation_id_for_chunk(result.document)
        docs_by_id[chunk_id] = result.document
        scores_by_id.setdefault(chunk_id, 0.0)
        scores_by_id[chunk_id] += vector_weight * normalized_vector.get(chunk_id, 0.0)
        scores_by_id[chunk_id] += keyword_weight * normalized_keyword.get(chunk_id, 0.0)

    ranked_ids = sorted(scores_by_id, key=lambda chunk_id: scores_by_id[chunk_id], reverse=True)
    return [docs_by_id[chunk_id] for chunk_id in ranked_ids[:top_k]]
