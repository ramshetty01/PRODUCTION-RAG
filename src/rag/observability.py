from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from src.rag.citations import citation_id_for_chunk
from src.rag.config import RuntimeSettings


LOGGER = logging.getLogger("rag.api")


def new_request_id() -> str:
    return str(uuid.uuid4())


@dataclass
class TraceEvent:
    request_id: str
    query: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    answer: str | None = None
    citations: list[str] = field(default_factory=list)
    latency_ms: float | None = None
    token_usage: dict | None = None
    error: str | None = None

    def to_log_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieval_scores": self.retrieval_scores,
            "answer": self.answer,
            "citations": self.citations,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage or {},
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_log_dict(), sort_keys=True)


def chunk_ids(chunks) -> list[str]:
    return [citation_id_for_chunk(chunk) for chunk in chunks]


def citation_ids(citations: list[dict]) -> list[str]:
    return [str(citation["id"]) for citation in citations]


@dataclass
class MetricsRegistry:
    request_count: int = 0
    status_counts: dict[int, int] = field(default_factory=dict)
    latency_ms_total: float = 0.0

    def record_request(self, status_code: int, latency_ms: float) -> None:
        self.request_count += 1
        self.status_counts[status_code] = self.status_counts.get(status_code, 0) + 1
        self.latency_ms_total += latency_ms

    def to_prometheus(self) -> str:
        lines = [
            "# HELP rag_api_requests_total Total API requests handled.",
            "# TYPE rag_api_requests_total counter",
            f"rag_api_requests_total {self.request_count}",
            "# HELP rag_api_request_status_total API requests by HTTP status code.",
            "# TYPE rag_api_request_status_total counter",
        ]
        for status_code in sorted(self.status_counts):
            count = self.status_counts[status_code]
            lines.append(f'rag_api_request_status_total{{status_code="{status_code}"}} {count}')
        lines.extend(
            [
                "# HELP rag_api_request_latency_ms_total Cumulative API request latency in milliseconds.",
                "# TYPE rag_api_request_latency_ms_total counter",
                f"rag_api_request_latency_ms_total {round(self.latency_ms_total, 3)}",
            ]
        )
        return "\n".join(lines) + "\n"


def structured_request_log(method: str, path: str, status_code: int, latency_ms: float, request_id: str) -> str:
    return json.dumps(
        {
            "event": "api_request",
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 3),
        },
        sort_keys=True,
    )


class OptionalOpenTelemetry:
    def __init__(self, enabled: bool = False, tracer: Any | None = None, reason: str = "disabled"):
        self.enabled = enabled
        self.tracer = tracer
        self.reason = reason

    @contextmanager
    def span(self, name: str, attributes: dict | None = None):
        if not self.enabled or self.tracer is None:
            yield None
            return
        with self.tracer.start_as_current_span(name) as span:
            for key, value in (attributes or {}).items():
                if value is not None:
                    span.set_attribute(key, value)
            yield span


def configure_opentelemetry(settings: RuntimeSettings) -> OptionalOpenTelemetry:
    if not settings.otel_enabled:
        return OptionalOpenTelemetry(reason="disabled")
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        LOGGER.warning("OpenTelemetry is enabled but optional packages are not installed")
        return OptionalOpenTelemetry(reason="missing-opentelemetry-packages")

    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    exporter_kwargs = {}
    if settings.otel_exporter_otlp_endpoint:
        exporter_kwargs["endpoint"] = settings.otel_exporter_otlp_endpoint
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(**exporter_kwargs)))
    trace.set_tracer_provider(provider)
    return OptionalOpenTelemetry(enabled=True, tracer=trace.get_tracer(settings.otel_service_name), reason="enabled")


@contextmanager
def trace_latency(event: TraceEvent):
    start = time.perf_counter()
    try:
        yield event
    except Exception as exc:
        event.error = type(exc).__name__
        raise
    finally:
        event.latency_ms = round((time.perf_counter() - start) * 1000, 3)
