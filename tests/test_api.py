from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.rag.api import routes


class FakeVectorStore:
    def similarity_search(self, query, k):
        return [
            Document(
                page_content="A runner executes jobs.",
                metadata={
                    "source": "/tmp/docs.pdf",
                    "page": 2,
                    "chunk_index": 3,
                    "chunk_id": "docs:p2:c3",
                },
            )
        ][:k]


def test_health_endpoint_returns_status():
    client = TestClient(routes.create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_returns_answer_citations_and_retrieval(monkeypatch):
    monkeypatch.setattr(routes, "load_vectorstore", lambda persist_dir: FakeVectorStore())
    client = TestClient(routes.create_app())

    response = client.post("/query", json={"query": "What does a runner do?", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A runner executes jobs. [docs:p2:c3]"
    assert body["citations"][0]["id"] == "docs:p2:c3"
    assert body["retrieval"] == {
        "top_k": 1,
        "returned_chunks": 1,
        "chunk_ids": ["docs:p2:c3"],
    }


def test_query_endpoint_returns_clean_json_errors(monkeypatch):
    def raise_error(_persist_dir):
        raise RuntimeError("vector store missing")

    monkeypatch.setattr(routes, "load_vectorstore", raise_error)
    client = TestClient(routes.create_app())

    response = client.post("/query", json={"query": "What does a runner do?"})

    assert response.status_code == 400
    assert response.json() == {"detail": "vector store missing"}
