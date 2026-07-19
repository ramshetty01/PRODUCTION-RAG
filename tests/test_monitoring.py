from src.rag.monitoring import (
    FeedbackEvent,
    append_feedback,
    draft_eval_cases_from_feedback,
    load_feedback,
    monitoring_metrics,
    promote_feedback_to_eval_case,
    write_draft_eval_cases,
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
    event = FeedbackEvent("req-1", "What failed?", "Weak answer", False, ["docs:p0:c1"], 30.0, note="Missing source.")

    case = promote_feedback_to_eval_case(event)

    assert case["id"] == "feedback-req-1"
    assert case["question"] == "What failed?"
    assert case["verified"] is False
    assert case["review_status"] == "needs_human_review"
    assert case["expected_evidence"] == "Missing source."
    assert case["source_request"]["request_id"] == "req-1"
    assert case["source_request"]["citations"] == ["docs:p0:c1"]
    assert case["source_request"]["helpful"] is False


def test_feedback_events_write_draft_eval_cases_for_human_review(tmp_path):
    output = tmp_path / "drafts" / "feedback-candidates.jsonl"
    events = [
        FeedbackEvent("bad-1", "Bad?", "Weak answer", False, [], note="Missing citation."),
        FeedbackEvent("good-1", "Good?", "Grounded answer", True, ["docs:p0:c1"]),
    ]

    cases = draft_eval_cases_from_feedback(events)
    count = write_draft_eval_cases(events, output)

    rows = [line for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert [case["source_request"]["helpful"] for case in cases] == [False, True]
    assert count == 2
    assert len(rows) == 2
    assert '"review_status": "needs_human_review"' in rows[0]
