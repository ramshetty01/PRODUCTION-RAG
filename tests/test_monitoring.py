from src.rag.monitoring import (
    FeedbackEvent,
    append_feedback,
    load_feedback,
    monitoring_metrics,
    promote_feedback_to_eval_case,
)


def test_feedback_events_round_trip_to_jsonl(tmp_path):
    path = tmp_path / "feedback.jsonl"
    event = FeedbackEvent(
        request_id="req-1",
        query="What is a runner?",
        answer="A runner executes jobs. [docs:p2:c3]",
        helpful=True,
        citations=["docs:p2:c3"],
        latency_ms=12.0,
    )

    append_feedback(event, path)

    loaded = load_feedback(path)
    assert len(loaded) == 1
    assert loaded[0].request_id == "req-1"
    assert loaded[0].helpful is True


def test_monitoring_metrics_track_feedback_quality():
    events = [
        FeedbackEvent("req-1", "q1", "answer [docs:p0:c0]", True, ["docs:p0:c0"], 10.0),
        FeedbackEvent("req-2", "q2", "The answer is not available in the retrieved context.", False, [], 30.0),
    ]

    metrics = monitoring_metrics(events)

    assert metrics["total"] == 2
    assert metrics["helpful_rate"] == 0.5
    assert metrics["no_answer_rate"] == 0.5
    assert metrics["citation_coverage"] == 0.5
    assert metrics["avg_latency_ms"] == 20.0


def test_feedback_can_be_promoted_to_eval_case_for_review():
    event = FeedbackEvent("req-1", "What failed?", "Weak answer", False, [], note="Missing source.")

    case = promote_feedback_to_eval_case(event)

    assert case["id"] == "feedback-req-1"
    assert case["question"] == "What failed?"
    assert case["verified"] is False
    assert case["expected_evidence"] == "Missing source."
