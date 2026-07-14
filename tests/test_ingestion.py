from langchain_core.documents import Document

from src.rag.vector_store import build_chroma_db, count_records, load_chroma_db


class FakeEmbeddings:
    def _embed(self, text):
        lower = text.lower()
        return [
            float(lower.count("workflow")),
            float(lower.count("runner")),
            float(lower.count("job")),
            float(len(text.split())),
        ]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_build_chroma_db_persists_one_record_per_chunk(tmp_path):
    chunks = [
        Document(
            page_content="A workflow is an automated process.",
            metadata={"source": "/tmp/docs.pdf", "page": 0, "chunk_index": 0, "chunk_id": "docs:p0:c0"},
        ),
        Document(
            page_content="A runner executes jobs.",
            metadata={"source": "/tmp/docs.pdf", "page": 1, "chunk_index": 1, "chunk_id": "docs:p1:c1"},
        ),
    ]
    persist_dir = tmp_path / "chroma_db"
    embeddings = FakeEmbeddings()

    vectorstore = build_chroma_db(chunks, persist_dir, embedding_function=embeddings)
    reloaded = load_chroma_db(persist_dir, embedding_function=embeddings)
    results = reloaded.similarity_search("runner job", k=1)

    assert persist_dir.exists()
    assert count_records(vectorstore) == len(chunks)
    assert count_records(reloaded) == len(chunks)
    assert results[0].metadata["chunk_id"] == "docs:p1:c1"
