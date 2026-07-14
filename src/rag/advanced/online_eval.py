from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from src.rag.monitoring import FeedbackEvent, promote_feedback_to_eval_case


NO_ANSWER_TEXT = "not available in the retrieved context"
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")


@dataclass(frozen=True)
class OnlineEvalScore:
    request_id: str
    groundedness: float
    citation_quality: float
    refusal_correctness: float
    user_satisfaction: float
    failed: bool
    reasons: list[str]


def redact_pii(text: str) -> str:
    text = EMAIL_PATTERN.sub("[redacted-email]", text)
    return PHONE_PATTERN.sub("[redacted-phone]", text)


def should_sample(request_id: str, sample_rate: float = 0.05) -> bool:
    if sample_rate <= 0:
        return False
    if sample_rate >= 1:
        return True
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < sample_rate


def sampled_events(events: list[FeedbackEvent], sample_rate: float = 0.05) -> list[FeedbackEvent]:
    return [event for event in events if should_sample(event.request_id, sample_rate)]


def score_online_event(event: FeedbackEvent) -> OnlineEvalScore:
    answer_lower = event.answer.lower()
    refused = NO_ANSWER_TEXT in answer_lower
    has_citations = bool(event.citations)
    reasons = []

    groundedness = 1.0 if refused or has_citations else 0.0
    citation_quality = 1.0 if has_citations else 0.0
    refusal_correctness = 1.0
    if refused and event.helpful:
        refusal_correctness = 0.5
        reasons.append("helpful refusal needs review")
    if not refused and not has_citations:
        reasons.append("answer missing citations")
    if not event.helpful:
        reasons.append("negative user feedback")

    user_satisfaction = 1.0 if event.helpful else 0.0
    failed = bool(reasons)
    return OnlineEvalScore(
        request_id=event.request_id,
        groundedness=groundedness,
        citation_quality=citation_quality,
        refusal_correctness=refusal_correctness,
        user_satisfaction=user_satisfaction,
        failed=failed,
        reasons=reasons,
    )


def online_eval_metrics(events: list[FeedbackEvent]) -> dict:
    scores = [score_online_event(event) for event in events]
    total = len(scores)
    if total == 0:
        return {
            "total": 0,
            "groundedness": 0.0,
            "citation_quality": 0.0,
            "refusal_correctness": 0.0,
            "user_satisfaction": 0.0,
            "failed": 0,
        }

    return {
        "total": total,
        "groundedness": sum(score.groundedness for score in scores) / total,
        "citation_quality": sum(score.citation_quality for score in scores) / total,
        "refusal_correctness": sum(score.refusal_correctness for score in scores) / total,
        "user_satisfaction": sum(score.user_satisfaction for score in scores) / total,
        "failed": sum(1 for score in scores if score.failed),
    }


def promote_failed_online_examples(events: list[FeedbackEvent]) -> list[dict]:
    promoted = []
    for event in events:
        score = score_online_event(event)
        if not score.failed:
            continue
        redacted = FeedbackEvent(
            request_id=event.request_id,
            query=redact_pii(event.query),
            answer=redact_pii(event.answer),
            helpful=event.helpful,
            citations=event.citations,
            latency_ms=event.latency_ms,
            note=f"Online eval failed: {', '.join(score.reasons)}",
            created_at=event.created_at,
        )
        case = promote_feedback_to_eval_case(redacted)
        case["online_eval"] = asdict(score)
        promoted.append(case)
    return promoted
