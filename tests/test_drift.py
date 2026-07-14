from src.rag.advanced.drift import calculate_drift_metrics, drift_triggers, query_distribution, retrieval_score_summary
from src.rag.monitoring import FeedbackEvent


def test_query_distribution_counts_production_query_terms():
    events = [
        FeedbackEvent("req-1", "What is a runner?", "answer [docs:p0:c0]", True, ["docs:p0:c0"]),
        FeedbackEvent("req-2", "What is a workflow?", "answer [docs:p0:c1]", True, ["docs:p0:c1"]),
    ]

    distribution = query_distribution(events)

    assert distribution["what"] == 2
    assert distribution["runner"] == 1
    assert distribution["workflow"] == 1


def test_drift_metrics_include_feedback_and_retrieval_score_signals():
    events = [
        FeedbackEvent("req-1", "q1", "answer [docs:p0:c0]", True, ["docs:p0:c0"], 10.0),
        FeedbackEvent("req-2", "q2", "The answer is not available in the retrieved context.", False, [], 20.0),
    ]

    metrics = calculate_drift_metrics(events, retrieval_scores=[0.9, 0.7])

    assert metrics["no_answer_rate"] == 0.5
    assert metrics["citation_coverage"] == 0.5
    assert metrics["helpful_rate"] == 0.5
    assert metrics["retrieval_scores"]["avg"] == 0.8


def test_drift_triggers_flag_threshold_breaches():
    metrics = {
        "no_answer_rate": 0.5,
        "citation_coverage": 0.4,
        "helpful_rate": 0.3,
    }

    assert drift_triggers(metrics) == ["no_answer_rate", "citation_coverage", "helpful_rate"]


def test_retrieval_score_summary_handles_empty_scores():
    assert retrieval_score_summary([]) == {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0}
