from __future__ import annotations

from collections import Counter

from src.rag.hybrid_search import tokenize
from src.rag.monitoring import FeedbackEvent, monitoring_metrics


DEFAULT_DRIFT_THRESHOLDS = {
    "no_answer_rate": 0.25,
    "citation_coverage_min": 0.80,
    "helpful_rate_min": 0.70,
    "retrieval_score_drop": 0.20,
}


def query_distribution(events: list[FeedbackEvent]) -> dict[str, int]:
    counts = Counter()
    for event in events:
        counts.update(tokenize(event.query))
    return dict(counts)


def retrieval_score_summary(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(scores),
        "avg": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
    }


def calculate_drift_metrics(
    events: list[FeedbackEvent],
    retrieval_scores: list[float] | None = None,
) -> dict:
    metrics = monitoring_metrics(events)
    return {
        **metrics,
        "query_distribution": query_distribution(events),
        "retrieval_scores": retrieval_score_summary(retrieval_scores or []),
    }


def drift_triggers(metrics: dict, thresholds: dict | None = None) -> list[str]:
    thresholds = thresholds or DEFAULT_DRIFT_THRESHOLDS
    triggers = []
    if metrics["no_answer_rate"] > thresholds["no_answer_rate"]:
        triggers.append("no_answer_rate")
    if metrics["citation_coverage"] < thresholds["citation_coverage_min"]:
        triggers.append("citation_coverage")
    if metrics["helpful_rate"] < thresholds["helpful_rate_min"]:
        triggers.append("helpful_rate")
    return triggers
