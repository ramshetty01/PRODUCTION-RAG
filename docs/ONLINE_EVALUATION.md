# Online Evaluation Pipeline

Online evaluation measures production traffic continuously so the team can find
quality regressions before they become offline golden-dataset failures.

## Metrics

- Groundedness: answers must either cite retrieved context or explicitly refuse.
- Citation quality: non-refusal answers should include citation ids.
- Refusal correctness: refusals are reviewed when users mark them helpful or
  unhelpful because both cases can reveal recall or expectation gaps.
- User satisfaction: direct feedback from production events.
- Failed examples: events with negative feedback, missing citations, or review
  needed for refusal behavior.

## Sampling

Production traces are sampled deterministically by request id. This keeps review
volume predictable and makes sampled examples reproducible. Use a low default
sample rate for normal traffic and increase it temporarily during incidents or
launches.

## Promotion to Offline Evals

Failed online examples can be promoted into golden-evaluation candidates. These
cases are marked `verified=false` and require human review before they become
quality-gate inputs. Promotion preserves request ids, citations, and failure
reasons so operators can trace the production behavior.

Export draft candidates from the feedback log:

```bash
python scripts/promote_feedback.py \
  --feedback logs/feedback.jsonl \
  --output evals/drafts/feedback-candidates.jsonl
```

Review the draft JSONL manually, verify evidence, redact sensitive content, and
only then copy approved rows into `evals/golden.jsonl` with `verified=true`.

## Privacy

Online examples may contain user data. Before promotion, obvious emails and
phone numbers are redacted. Do not add promoted cases to `evals/golden.jsonl`
until a reviewer confirms the example is safe, useful, and supported by
available source evidence.

## Current Implementation

`src/rag/advanced/online_eval.py` provides deterministic sampling, lightweight
quality scoring, PII redaction, aggregate metrics, and promotion helpers. The
implementation uses existing `FeedbackEvent` traces so it can run locally or in
a scheduled production job without adding a new service.
