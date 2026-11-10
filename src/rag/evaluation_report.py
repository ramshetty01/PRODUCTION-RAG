from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from evals.run_ragas import (
    DEFAULT_CONFIG,
    DEFAULT_DATASET,
    evaluate_dataset_by_mode,
    load_dataset,
    load_eval_mode,
    load_quality_threshold,
    quality_gate,
)


def _category_counts(cases: list[dict]) -> dict:
    return dict(sorted(Counter(case.get("category", "uncategorized") for case in cases).items()))


def build_evaluation_report(
    dataset_path: str | Path = DEFAULT_DATASET,
    config_path: str | Path = DEFAULT_CONFIG,
    mode: str | None = None,
    predictions: list[dict] | None = None,
) -> dict:
    dataset_path = Path(dataset_path)
    config_path = Path(config_path)
    cases = load_dataset(dataset_path)
    eval_mode = mode or load_eval_mode(config_path)
    min_faithfulness = load_quality_threshold(config_path)
    metrics = evaluate_dataset_by_mode(cases, mode=eval_mode, predictions=predictions)
    passed, message = quality_gate(metrics, min_faithfulness)

    latency_budget_ms = 2000
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "total_cases": metrics["total_cases"],
            "verified_cases": metrics["verified_cases"],
            "category_counts": _category_counts(cases),
        },
        "config": {
            "path": str(config_path),
            "mode": metrics.get("mode", eval_mode),
            "min_faithfulness": min_faithfulness,
            "retrieval_latency_budget_ms": latency_budget_ms,
        },
        "metrics": {
            "faithfulness": metrics["faithfulness"],
            "context_precision": metrics["context_precision"],
            "answer_relevance": metrics["answer_relevance"],
            "citation_coverage": metrics["citation_coverage"],
            "refusal_accuracy": metrics["refusal_accuracy"],
            "latency_budget_pass": metrics["latency_budget_pass"],
            "lexical_support_proxy": metrics["lexical_support_proxy"],
            "reference_token_recall": metrics["reference_token_recall"],
        },
        "quality_gate": {
            "passed": passed if metrics["faithfulness"] is not None else None,
            "status": "evaluated" if metrics["faithfulness"] is not None else "not_evaluated",
            "message": message,
        },
        "prediction_cases": metrics["prediction_cases"],
        "note": metrics["note"],
        "case_scores": metrics.get("case_scores", []),
    }
