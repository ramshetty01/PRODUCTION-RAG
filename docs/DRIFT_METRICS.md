# Drift Metrics

The project tracks drift from production feedback and retrieval traces.

## Signals

- Query distribution: token counts from production questions.
- Retrieval scores: count, average, minimum, and maximum retrieval scores.
- No-answer rate: share of answers that refused due to missing context.
- Citation coverage: share of answers with citations.
- Helpful rate: share of feedback marked helpful.

## Review Triggers

Configured in `configs/settings.toml`:

- `no_answer_rate`: trigger when refusals exceed the threshold.
- `citation_coverage_min`: trigger when citation coverage falls below the
  minimum.
- `helpful_rate_min`: trigger when helpful feedback falls below the minimum.
- `retrieval_score_drop`: reserved for comparing current retrieval score
  averages against a baseline.

These signals are intentionally lightweight so they can run locally and in logs
before a full production metrics backend exists.
