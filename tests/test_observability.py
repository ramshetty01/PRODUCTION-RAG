import json

import pytest
from langchain_core.documents import Document

from src.rag.observability import TraceEvent, chunk_ids, citation_ids, new_request_id, trace_latency


def test_trace_event_serializes_structured_log_fields():
    event = TraceEvent(
        request_id="req-1",
        query="What is a runner?",
        retrieved_chunk_ids=["docs:p2:c3"],
        retrieval_scores=[0.9],
        answer="A runner executes jobs. [docs:p2:c3]",
        citations=["docs:p2:c3"],
        latency_ms=12.3,
        token_usage={"prompt_tokens": 10, "answer_tokens": 5},
    )

    payload = json.loads(event.to_json())

    assert payload["request_id"] == "req-1"
    assert payload["retrieved_chunk_ids"] == ["docs:p2:c3"]
    assert payload["token_usage"] == {"prompt_tokens": 10, "answer_tokens": 5}
    assert payload["error"] is None


def test_trace_helpers_extract_chunk_and_citation_ids():
    chunk = Document(page_content="text", metadata={"chunk_id": "docs:p0:c0"})

    assert chunk_ids([chunk]) == ["docs:p0:c0"]
    assert citation_ids([{"id": "docs:p0:c0"}]) == ["docs:p0:c0"]
    assert new_request_id()


def test_trace_latency_records_errors_and_duration():
    event = TraceEvent(request_id="req-1", query="bad query")

    with pytest.raises(RuntimeError):
        with trace_latency(event):
            raise RuntimeError("boom")

    assert event.error == "RuntimeError"
    assert event.latency_ms >= 0
