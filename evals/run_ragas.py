"""Validate reference fixtures or evaluate explicit recorded pipeline predictions."""
from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path

DEFAULT_DATASET = Path(__file__).resolve().parent / "golden.jsonl"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "settings.toml"
DEFAULT_EVAL_MODE = "deterministic"


def _normalize(text: str) -> set[str]:
    return {token.strip(".,;:!?()[]").lower() for token in text.split() if token.strip(".,;:!?()[]")}


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_dataset(cases: list[dict]) -> None:
    if any(not isinstance(case, dict) for case in cases):
        raise ValueError("Reference cases must be JSON objects.")
    ids = [case.get("id") for case in cases]
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Reference cases need nonempty unique IDs.")
    for case in cases:
        if not all(isinstance(case.get(key), str) and case[key] for key in ("question", "expected_answer", "expected_evidence")):
            raise ValueError("Reference cases require question, expected answer and evidence strings.")


def prediction_index(cases: list[dict], predictions: list[dict] | None) -> dict:
    valid_ids = {case["id"] for case in cases}
    result = {}
    for row in predictions or []:
        if not isinstance(row, dict):
            raise ValueError("Predictions must be JSON objects.")
        case_id = row.get("id")
        if case_id not in valid_ids or case_id in result:
            raise ValueError("Predictions contain unknown or duplicate case IDs.")
        if not isinstance(row.get("answer"), str):
            raise ValueError("Each prediction requires the actual answer string.")
        for field in ("contexts", "citations"):
            if not isinstance(row.get(field), list) or not all(isinstance(value, str) for value in row[field]):
                raise ValueError(f"Each prediction requires actual {field} strings.")
        latency = row.get("retrieval_ms")
        if latency is not None and (type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0):
            raise ValueError("Prediction retrieval_ms must be a finite nonnegative number.")
        if "refused" in row and type(row["refused"]) is not bool:
            raise ValueError("Prediction refused must be a boolean.")
        result[case_id] = row
    return result


def score_case(case: dict, prediction: dict | None = None) -> dict:
    # Never substitute expected_answer/evidence for actual inference outputs.
    answer_terms = _normalize(prediction["answer"]) if prediction else set()
    context_terms = _normalize(" ".join(prediction["contexts"])) if prediction else set()
    reference_terms = _normalize(case["expected_answer"])
    return {
        "id": case["id"],
        "prediction_available": prediction is not None,
        "lexical_support_proxy": len(answer_terms & context_terms) / len(answer_terms) if answer_terms else 0.0,
        "reference_token_recall": len(reference_terms & answer_terms) / len(reference_terms) if reference_terms else 0.0,
        "citation_present": float(bool(prediction and prediction["citations"])),
    }


def evaluate_dataset(cases: list[dict], predictions: list[dict] | None = None) -> dict:
    validate_dataset(cases)
    indexed = prediction_index(cases, predictions)
    scores = [score_case(case, indexed.get(case["id"])) for case in cases]
    supplied = predictions is not None
    refusal_cases = [c for c in cases if c.get("category") == "refusal"]
    measured_latencies = [row["retrieval_ms"] for row in indexed.values() if row.get("retrieval_ms") is not None]
    return {
        "total_cases": len(cases),
        "verified_cases": sum(c.get("verified") is True for c in cases),
        "mode": "deterministic",
        "status": "predictions_scored" if supplied else "not_run",
        "prediction_cases": len(indexed),
        "missing_prediction_cases": len(cases) - len(indexed),
        "faithfulness": None,
        "context_precision": None,
        "answer_relevance": None,
        "citation_coverage": sum(row["citation_present"] for row in scores) / len(cases) if supplied else None,
        "lexical_support_proxy": sum(row["lexical_support_proxy"] for row in scores) / len(cases) if supplied else None,
        "reference_token_recall": sum(row["reference_token_recall"] for row in scores) / len(cases) if supplied else None,
        "refusal_accuracy": sum(indexed.get(c["id"], {}).get("refused") is True for c in refusal_cases) / len(refusal_cases) if supplied and refusal_cases else None,
        "latency_budget_pass": all(value <= 2000 for value in measured_latencies) if len(measured_latencies) == len(cases) else None,
        "case_scores": scores if supplied else [],
        "note": "Reference verified flags are fixture metadata, not proof of independent review. Lexical overlap and citation presence are diagnostic proxies, not semantic faithfulness/correctness. Missing predictions stay in denominators. No pipeline quality is measured without actual predictions.",
    }


def load_quality_threshold(config_path: str | Path = DEFAULT_CONFIG) -> float:
    config = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    threshold = float(config["evaluation"]["min_faithfulness"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Faithfulness threshold must be between zero and one.")
    return threshold


def load_eval_mode(config_path: str | Path = DEFAULT_CONFIG) -> str:
    config = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    return str(config.get("evaluation", {}).get("mode", DEFAULT_EVAL_MODE))


def _ragas_dataset_rows(cases: list[dict], predictions: list[dict]) -> list[dict]:
    validate_dataset(cases)
    indexed = prediction_index(cases, predictions)
    if len(indexed) != len(cases):
        raise ValueError("Ragas requires an actual prediction for every reference case.")
    return [{"question": c["question"], "answer": indexed[c["id"]]["answer"], "contexts": indexed[c["id"]]["contexts"], "ground_truth": c["expected_answer"], "reference": c["expected_answer"]} for c in cases]


def evaluate_dataset_with_ragas(cases: list[dict], predictions: list[dict] | None = None) -> dict:
    if predictions is None:
        raise ValueError("Ragas requires actual recorded predictions; expected answers cannot substitute.")
    rows = _ragas_dataset_rows(cases, predictions)
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        raise RuntimeError("Ragas requires optional ragas/datasets dependencies and explicit judge credentials.") from exc
    result = evaluate(Dataset.from_list(rows), metrics=[faithfulness, answer_relevancy, context_precision])
    scores = dict(result)
    report = evaluate_dataset(cases, predictions)
    report.update(mode="ragas", faithfulness=float(scores["faithfulness"]), context_precision=float(scores["context_precision"]), answer_relevance=float(scores["answer_relevancy"]))
    if not all(math.isfinite(report[name]) and 0 <= report[name] <= 1 for name in ("faithfulness", "context_precision", "answer_relevance")):
        raise ValueError("Ragas returned unavailable or invalid metrics; quality gate cannot pass.")
    report["note"] += " Ragas judgments are model-based and require independent validation."
    return report


def evaluate_dataset_by_mode(cases: list[dict], mode: str = DEFAULT_EVAL_MODE, predictions: list[dict] | None = None) -> dict:
    if mode.lower() in {"deterministic", "auto"}:
        # Auto never starts a paid judge or silently replaces a failed judge with a passing proxy.
        return evaluate_dataset(cases, predictions)
    if mode.lower() == "ragas":
        return evaluate_dataset_with_ragas(cases, predictions)
    raise ValueError(f"Unsupported evaluation mode: {mode}")


def quality_gate(metrics: dict, min_faithfulness: float) -> tuple[bool, str]:
    value = metrics.get("faithfulness")
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        return False, "faithfulness not measured; quality gate not evaluated"
    if value >= min_faithfulness:
        return True, f"faithfulness passed: {value:.2f} >= {min_faithfulness:.2f}"
    return False, f"faithfulness failed: {value:.2f} < {min_faithfulness:.2f}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--predictions", help="JSONL actual outputs: id, answer, contexts, citations; optional refused/retrieval_ms")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--min-faithfulness", type=float)
    parser.add_argument("--mode", choices=["deterministic", "ragas", "auto"])
    parser.add_argument("--require-quality-gate", action="store_true", help="Fail unless semantic faithfulness was measured and passed")
    parser.add_argument("--output", help="Optional complete JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    threshold = args.min_faithfulness if args.min_faithfulness is not None else load_quality_threshold(args.config)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Faithfulness threshold must be between zero and one.")
    predictions = load_dataset(args.predictions) if args.predictions else None
    metrics = evaluate_dataset_by_mode(load_dataset(args.dataset), mode=args.mode or load_eval_mode(args.config), predictions=predictions)
    passed, message = quality_gate(metrics, threshold)
    print(json.dumps(metrics, indent=2))
    print(message)
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if args.require_quality_gate or metrics["mode"] == "ragas":
        return 0 if passed else 1
    print("Fixture validation/diagnostics only; this exit status is not a model-quality gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
