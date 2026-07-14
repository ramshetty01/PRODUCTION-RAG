from langchain_core.documents import Document

from src.rag.chunking import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_TOKENS,
    chunk_documents,
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


def test_neighboring_chunks_keep_overlap():
    text = " ".join(f"word{i}" for i in range(1200))
    docs = [Document(page_content=text, metadata={"source": "docs.pdf", "page": 2})]

    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=DEFAULT_CHUNK_OVERLAP_TOKENS)

    first_tokens = chunks[0].page_content.split()
    second_tokens = chunks[1].page_content.split()

    assert first_tokens[-DEFAULT_CHUNK_OVERLAP_TOKENS:] == second_tokens[
        :DEFAULT_CHUNK_OVERLAP_TOKENS
    ]
