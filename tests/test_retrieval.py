from langchain_core.documents import Document

from src.rag.generation import generate_answer
from src.rag.retrieval import retrieve_chunks


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def similarity_search(self, query, k):
        self.calls.append({"query": query, "k": k})
        return self.docs[:k]


class RecordingLLM:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.answer


def make_doc(text, chunk_id):
    return Document(
        page_content=text,
        metadata={
            "source": "/tmp/docs.pdf",
            "page": 1,
            "chunk_index": 2,
            "chunk_id": chunk_id,
        },
    )


def test_retrieve_chunks_uses_vector_similarity_and_top_k():
    docs = [make_doc("first", "docs:p1:c0"), make_doc("second", "docs:p1:c1")]
    vectorstore = FakeVectorStore(docs)

    results = retrieve_chunks("What is a job?", vectorstore, top_k=1)

    assert results == [docs[0]]
    assert vectorstore.calls == [{"query": "What is a job?", "k": 1}]


def test_generate_answer_sends_query_and_context_to_llm_with_citations():
    chunks = [
        make_doc("A job is a set of steps in a workflow.", "docs:p1:c2"),
        make_doc("A runner executes jobs.", "docs:p2:c3"),
    ]
    llm = RecordingLLM("A job is a set of steps in a workflow. [docs:p1:c2]")

    response = generate_answer("What is a job?", chunks, llm=llm)

    assert "Question: What is a job?" in llm.prompts[0]
    assert "[docs:p1:c2]" in llm.prompts[0]
    assert "A runner executes jobs." in llm.prompts[0]
    assert response["answer"] == "A job is a set of steps in a workflow. [docs:p1:c2]"
    assert [citation["id"] for citation in response["citations"]] == ["docs:p1:c2"]
