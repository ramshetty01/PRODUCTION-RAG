from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.rag.citations import citation_id_for_chunk
from src.rag.hybrid_search import SearchResult, tokenize


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0


@dataclass
class DocumentGraph:
    documents_by_id: dict[str, object]
    edges_by_source: dict[str, list[GraphEdge]] = field(default_factory=dict)

    def neighbors(self, chunk_id: str) -> list[GraphEdge]:
        return self.edges_by_source.get(chunk_id, [])


def _metadata_key(document, field: str) -> str | None:
    value = document.metadata.get(field)
    if value in (None, ""):
        return None
    return f"{field}:{value}"


def build_document_graph(
    documents,
    metadata_fields: list[str] | None = None,
    min_shared_terms: int = 2,
) -> DocumentGraph:
    metadata_fields = metadata_fields or ["document_id", "source", "section"]
    docs = list(documents)
    documents_by_id = {citation_id_for_chunk(document): document for document in docs}
    edges_by_source: dict[str, list[GraphEdge]] = defaultdict(list)

    buckets: dict[str, list[str]] = defaultdict(list)
    token_sets: dict[str, set[str]] = {}
    for document in docs:
        chunk_id = citation_id_for_chunk(document)
        token_sets[chunk_id] = set(tokenize(document.page_content))
        for field in metadata_fields:
            key = _metadata_key(document, field)
            if key:
                buckets[key].append(chunk_id)

    for key, chunk_ids in buckets.items():
        relation = key.split(":", 1)[0]
        for source_id in chunk_ids:
            for target_id in chunk_ids:
                if source_id != target_id:
                    edges_by_source[source_id].append(GraphEdge(source_id, target_id, relation, 1.0))

    chunk_ids = list(documents_by_id)
    for index, source_id in enumerate(chunk_ids):
        for target_id in chunk_ids[index + 1 :]:
            shared_terms = token_sets[source_id] & token_sets[target_id]
            if len(shared_terms) < min_shared_terms:
                continue
            weight = min(1.0, len(shared_terms) / 10)
            edges_by_source[source_id].append(GraphEdge(source_id, target_id, "shared_terms", weight))
            edges_by_source[target_id].append(GraphEdge(target_id, source_id, "shared_terms", weight))

    return DocumentGraph(documents_by_id=documents_by_id, edges_by_source=dict(edges_by_source))


def graph_expand(seed_documents, graph: DocumentGraph, limit: int = 4) -> list[SearchResult]:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    seen = {citation_id_for_chunk(document) for document in seed_documents}
    results: list[SearchResult] = []
    for seed in seed_documents:
        seed_id = citation_id_for_chunk(seed)
        for edge in graph.neighbors(seed_id):
            if edge.target_id in seen:
                continue
            target = graph.documents_by_id.get(edge.target_id)
            if target is None:
                continue
            seen.add(edge.target_id)
            results.append(SearchResult(document=target, score=edge.weight, source=f"graph:{edge.relation}"))

    results.sort(key=lambda result: result.score, reverse=True)
    return results[:limit]


def graph_retrieve(query: str, documents, seed_search, top_k: int = 4) -> list[SearchResult]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    seed_results = seed_search(query, documents, top_k)
    seed_documents = [result.document for result in seed_results]
    graph = build_document_graph(documents)
    expanded = graph_expand(seed_documents, graph, limit=top_k)
    combined = [*seed_results, *expanded]

    deduped = []
    seen = set()
    for result in combined:
        chunk_id = citation_id_for_chunk(result.document)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(result)
    return deduped[:top_k]


def graph_rag_decision(documents) -> str:
    graph = build_document_graph(documents)
    edge_count = sum(len(edges) for edges in graph.edges_by_source.values())
    if edge_count == 0:
        return "no-build: documents do not expose useful relationships yet"
    if edge_count < len(graph.documents_by_id):
        return "explore: sparse relationships exist, validate before production"
    return "prototype: relationships are dense enough for graph expansion experiments"
