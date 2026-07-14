from src.rag.advanced.online_eval import (
    online_eval_metrics,
    promote_failed_online_examples,
    redact_pii,
    sampled_events,
    score_online_event,
    should_sample,
)
from src.rag.monitoring import FeedbackEvent


def test_online_eval_scores_missing_citations_as_failed():
    event = FeedbackEvent("req-1", "What is a workflow?", "A workflow runs jobs.", True, [])

    score = score_online_event(event)

    assert score.failed is True
    assert score.groundedness == 0.0
    assert "answer missing citations" in score.reasons


def test_online_eval_metrics_aggregate_quality_signals():
    events = [
        FeedbackEvent("req-1", "q1", "answer [docs:p0:c0]", True, ["docs:p0:c0"]),
        FeedbackEvent("req-2", "q2", "answer without citation", False, []),
    ]

    metrics = online_eval_metrics(events)

    assert metrics["total"] == 2
    assert metrics["groundedness"] == 0.5
    assert metrics["citation_quality"] == 0.5
    assert metrics["user_satisfaction"] == 0.5
    assert metrics["failed"] == 1


def test_sampling_is_deterministic_for_request_id():
    assert should_sample("same-request", 0.5) == should_sample("same-request", 0.5)
    assert sampled_events([FeedbackEvent("req-1", "q", "a", True, [])], sample_rate=0) == []


def test_failed_online_examples_are_redacted_and_promoted():
    event = FeedbackEvent(
        "req-9",
        "Email me at user@example.com about 555-123-4567",
        "answer without citation",
        False,
        [],
    )

    promoted = promote_failed_online_examples([event])

    assert promoted[0]["verified"] is False
    assert "[redacted-email]" in promoted[0]["question"]
    assert "[redacted-phone]" in promoted[0]["question"]
    assert promoted[0]["online_eval"]["failed"] is True


def test_redact_pii_handles_email_and_phone_text():
    assert redact_pii("a@b.com +1 555 123 4567") == "[redacted-email] [redacted-phone]"
