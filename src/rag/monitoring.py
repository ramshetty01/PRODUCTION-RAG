from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_FEEDBACK_LOG = Path("logs/feedback.jsonl")


@dataclass
class FeedbackEvent:
    request_id: str
    query: str
    answer: str
    helpful: bool
    citations: list[str]
    latency_ms: float | None = None
    note: str | None = None
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()


def append_feedback(event: FeedbackEvent, path: str | Path = DEFAULT_FEEDBACK_LOG) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")


def load_feedback(path: str | Path = DEFAULT_FEEDBACK_LOG) -> list[FeedbackEvent]:
    path = Path(path)
    if not path.exists():
        return []
    return [FeedbackEvent(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def monitoring_metrics(events: list[FeedbackEvent]) -> dict:
    total = len(events)
    if total == 0:
        return {"total": 0, "helpful_rate": 0.0, "no_answer_rate": 0.0, "citation_coverage": 0.0, "avg_latency_ms": 0.0}
    helpful = sum(1 for event in events if event.helpful)
    no_answer = sum(1 for event in events if "not available in the retrieved context" in event.answer.lower())
    cited = sum(1 for event in events if event.citations)
    latencies = [event.latency_ms for event in events if event.latency_ms is not None]
    return {
        "total": total,
        "helpful_rate": helpful / total,
        "no_answer_rate": no_answer / total,
        "citation_coverage": cited / total,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def promote_feedback_to_eval_case(event: FeedbackEvent) -> dict:
    return {
        "id": f"feedback-{event.request_id}",
        "question": event.query,
        "expected_answer": event.answer,
        "expected_citations": event.citations,
        "expected_evidence": event.note or "Needs manual review from production feedback.",
        "source": "production-feedback",
        "page": 0,
        "verified": False,
    }
