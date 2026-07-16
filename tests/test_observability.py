import json

import pytest
from langchain_core.documents import Document

from src.rag.observability import (
    MetricsRegistry,
    OptionalOpenTelemetry,
    TraceEvent,
    chunk_ids,
    citation_ids,
    configure_opentelemetry,
    new_request_id,
    structured_request_log,
    trace_latency,
)
from src.rag.config import RuntimeSettings


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


def test_metrics_registry_exports_prometheus_text():
    metrics = MetricsRegistry()

    metrics.record_request(200, 12.3456)
    metrics.record_request(400, 4.0)
    payload = metrics.to_prometheus()

    assert "rag_api_requests_total 2" in payload
    assert 'rag_api_request_status_total{status_code="200"} 1' in payload
    assert 'rag_api_request_status_total{status_code="400"} 1' in payload
    assert "rag_api_request_latency_ms_total 16.346" in payload


def test_structured_request_log_is_machine_parseable():
    payload = json.loads(structured_request_log("GET", "/health", 200, 1.23456, "req-1"))

    assert payload == {
        "event": "api_request",
        "latency_ms": 1.235,
        "method": "GET",
        "path": "/health",
        "request_id": "req-1",
        "status_code": 200,
    }


def test_optional_opentelemetry_span_sets_attributes():
    class FakeSpan:
        def __init__(self):
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

    class FakeSpanContext:
        def __init__(self, span):
            self.span = span

        def __enter__(self):
            return self.span

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeTracer:
        def __init__(self):
            self.started = []

        def start_as_current_span(self, name):
            span = FakeSpan()
            self.started.append((name, span))
            return FakeSpanContext(span)

    tracer = FakeTracer()
    telemetry = OptionalOpenTelemetry(enabled=True, tracer=tracer, reason="test")

    with telemetry.span("rag.retrieval", {"rag.request_id": "req-1", "rag.top_k": 4, "skip": None}):
        pass

    assert tracer.started[0][0] == "rag.retrieval"
    assert tracer.started[0][1].attributes == {"rag.request_id": "req-1", "rag.top_k": 4}


def test_opentelemetry_config_is_noop_when_disabled():
    telemetry = configure_opentelemetry(RuntimeSettings(otel_enabled=False))

    assert telemetry.enabled is False
    assert telemetry.reason == "disabled"
