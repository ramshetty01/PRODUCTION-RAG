from langchain_core.documents import Document

from src.rag.advanced.graph_rag import (
    build_document_graph,
    graph_expand,
    graph_rag_decision,
    graph_retrieve,
)
from src.rag.hybrid_search import SearchResult


def make_doc(text, chunk_id, **metadata):
    base_metadata = {
        "source": "docs.pdf",
        "page": 1,
        "chunk_index": 1,
        "chunk_id": chunk_id,
        "document_id": "docs",
        "access_roles": ["public"],
    }
    base_metadata.update(metadata)
    return Document(page_content=text, metadata=base_metadata)


def lexical_seed_search(query, documents, top_k):
    matches = [
        document
        for document in documents
        if any(token.lower() in document.page_content.lower() for token in query.split())
    ]
    return [SearchResult(document=document, score=float(top_k - index), source="seed") for index, document in enumerate(matches[:top_k])]


def test_build_document_graph_links_related_metadata():
    first = make_doc("A workflow triggers a job.", "docs:p0:c0", section="workflow")
    second = make_doc("A job runs on a runner.", "docs:p0:c1", section="workflow")

    graph = build_document_graph([first, second])

    neighbor_ids = {edge.target_id for edge in graph.neighbors("docs:p0:c0")}
    assert "docs:p0:c1" in neighbor_ids


def test_graph_expand_adds_related_context_from_seed_document():
    seed = make_doc("Error ZX-144 appears in deployment.", "docs:p0:c0", section="incident")
    recovery = make_doc("Recovery requires rebuilding deployment cache.", "docs:p0:c1", section="incident")
    graph = build_document_graph([seed, recovery])

    expanded = graph_expand([seed], graph, limit=1)

    assert expanded[0].document == recovery
    assert expanded[0].source.startswith("graph:")


def test_graph_retrieve_combines_seed_and_graph_results_without_duplicates():
    seed = make_doc("Error ZX-144 appears in deployment.", "docs:p0:c0", section="incident")
    recovery = make_doc("Recovery requires rebuilding deployment cache.", "docs:p0:c1", section="incident")

    results = graph_retrieve("ZX-144", [seed, recovery], lexical_seed_search, top_k=2)

    assert [result.document.metadata["chunk_id"] for result in results] == ["docs:p0:c0", "docs:p0:c1"]


def test_graph_rag_decision_stays_exploratory_for_related_documents():
    docs = [
        make_doc("A workflow triggers a job.", "docs:p0:c0"),
        make_doc("A job runs on a runner.", "docs:p0:c1"),
    ]

    assert graph_rag_decision(docs).startswith("prototype:")
