from langchain_core.documents import Document

from src.rag.chunking import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_TOKENS,
    chunk_documents,
    chunk_text_file,
    chunk_token_summary,
    count_tokens,
)


def test_chunk_documents_uses_token_target_and_preserves_metadata():
    text = " ".join(f"token{i}" for i in range(2200))
    docs = [Document(page_content=text, metadata={"source": "docs.pdf", "page": 0})]

    chunks = chunk_documents(docs)

    assert len(chunks) > 1
    assert all(count_tokens(chunk.page_content) <= DEFAULT_CHUNK_TOKENS for chunk in chunks)
    assert chunks[0].metadata["source"].endswith("docs.pdf")
    assert chunks[0].metadata["page"] == 0
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_id"].startswith("docs:p0:c0")
    assert chunks[0].metadata["document_id"] == "docs"
    assert chunks[0].metadata["document_version"] == "v1"
    assert chunks[0].metadata["access_roles"] == ["public"]
    assert 500 <= DEFAULT_CHUNK_TOKENS <= 800
    assert DEFAULT_CHUNK_OVERLAP_TOKENS == 100


def test_neighboring_chunks_keep_overlap():
    text = " ".join(f"word{i}" for i in range(1200))
    docs = [Document(page_content=text, metadata={"source": "docs.pdf", "page": 2})]

    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=DEFAULT_CHUNK_OVERLAP_TOKENS)

    first_tokens = chunks[0].page_content.split()
    second_tokens = chunks[1].page_content.split()

    assert first_tokens[-DEFAULT_CHUNK_OVERLAP_TOKENS:] == second_tokens[
        :DEFAULT_CHUNK_OVERLAP_TOKENS
    ]


def test_chunk_token_summary_reports_verification_counts():
    text = " ".join(f"token{i}" for i in range(900))
    docs = [Document(page_content=text, metadata={"source": "docs.pdf", "page": 0})]

    chunks = chunk_documents(docs)
    summary = chunk_token_summary(chunks)

    assert summary["chunks"] == len(chunks)
    assert summary["max_tokens"] <= DEFAULT_CHUNK_TOKENS
    assert summary["target_tokens"] == DEFAULT_CHUNK_TOKENS
    assert summary["overlap_tokens"] == DEFAULT_CHUNK_OVERLAP_TOKENS


def test_chunk_text_file_ingests_markdown_with_document_metadata(tmp_path):
    source = tmp_path / "security-policy.md"
    source.write_text("# Security Policy\n\nAll vendors require SOC 2 evidence.", encoding="utf-8")

    chunks = chunk_text_file(source, chunk_size=12, chunk_overlap=2, document_version="v3")

    assert chunks
    assert chunks[0].metadata["source"] == str(source.resolve())
    assert chunks[0].metadata["page"] == 0
    assert chunks[0].metadata["document_id"] == "security-policy"
    assert chunks[0].metadata["document_version"] == "v3"
    assert chunks[0].metadata["access_roles"] == ["public"]
