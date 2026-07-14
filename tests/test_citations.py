from langchain_core.documents import Document

from src.rag.citations import (
    attach_citations,
    citation_for_chunk,
    format_context_with_citations,
    select_citations,
)


def make_chunk(text, chunk_id, page=0):
    return Document(
        page_content=text,
        metadata={
            "source": "/tmp/docs.pdf",
            "page": page,
            "chunk_index": int(chunk_id.rsplit("c", 1)[-1]),
            "chunk_id": chunk_id,
        },
    )


def test_citation_records_locate_source_page_and_chunk():
    chunk = make_chunk("GitHub Actions runs workflows on runners.", "docs:p0:c0", page=0)

    citation = citation_for_chunk(chunk)

    assert citation["id"] == "docs:p0:c0"
    assert citation["source"] == "docs.pdf"
    assert citation["source_path"] == "/tmp/docs.pdf"
    assert citation["page"] == 0
    assert citation["chunk_index"] == 0
    assert citation["quote"] == "GitHub Actions runs workflows on runners."


def test_context_includes_citation_ids_and_metadata_for_prompting():
    chunks = [
        make_chunk("A workflow is an automated process.", "docs:p0:c0", page=0),
        make_chunk("A job is a set of steps.", "docs:p1:c1", page=1),
    ]

    context = format_context_with_citations(chunks)

    assert "[docs:p0:c0]" in context
    assert "source: docs.pdf" in context
    assert "page: 1" in context
    assert "A job is a set of steps." in context


def test_attach_citations_only_returns_retrieved_chunks():
    retrieved = [
        make_chunk("Workflow evidence.", "docs:p0:c0"),
        make_chunk("Runner evidence.", "docs:p1:c1", page=1),
    ]

    citations = select_citations(retrieved, cited_ids=["docs:p1:c1", "not-retrieved"])
    response = attach_citations("Runners execute jobs. [docs:p1:c1]", retrieved, ["docs:p1:c1"])

    assert [citation["id"] for citation in citations] == ["docs:p1:c1"]
    assert response["answer"] == "Runners execute jobs. [docs:p1:c1]"
    assert [citation["id"] for citation in response["citations"]] == ["docs:p1:c1"]
