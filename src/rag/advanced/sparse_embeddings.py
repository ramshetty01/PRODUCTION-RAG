from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from src.rag.citations import citation_id_for_chunk
from src.rag.hybrid_search import SearchResult, tokenize


SparseVector = dict[str, float]


@dataclass(frozen=True)
class SparseEmbeddingSettings:
    enabled: bool = False
    top_k: int = 4


class TfidfSparseEncoder:
    """Small local sparse encoder for lexical retrieval experiments."""

    def __init__(self, documents):
        self.documents = list(documents)
        self.document_tokens = [tokenize(document.page_content) for document in self.documents]
        self.document_frequencies = Counter()
        for tokens in self.document_tokens:
            self.document_frequencies.update(set(tokens))

    def idf(self, token: str) -> float:
        corpus_size = len(self.documents)
        frequency = self.document_frequencies.get(token, 0)
        return math.log((1 + corpus_size) / (1 + frequency)) + 1.0

    def encode_text(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return {token: (count / total) * self.idf(token) for token, count in counts.items()}

    def encode_document(self, index: int) -> SparseVector:
        return self.encode_text(self.documents[index].page_content)


def sparse_dot(left: SparseVector, right: SparseVector) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(token, 0.0) for token, weight in left.items())


def sparse_search(query: str, documents, top_k: int = 4) -> list[SearchResult]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    encoder = TfidfSparseEncoder(documents)
    query_vector = encoder.encode_text(query)
    scored = []
    for index, document in enumerate(encoder.documents):
        score = sparse_dot(query_vector, encoder.encode_document(index))
        if score > 0:
            scored.append(SearchResult(document=document, score=score, source="sparse_tfidf"))

    scored.sort(
        key=lambda result: (result.score, citation_id_for_chunk(result.document)),
        reverse=True,
    )
    return scored[:top_k]
