from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


DEFAULT_DATASET = Path(__file__).resolve().parent / "golden.jsonl"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "settings.toml"
DEFAULT_EVAL_MODE = "deterministic"


def _normalize(text: str) -> set[str]:
    return {token.strip(".,;:!?()[]").lower() for token in text.split() if token.strip(".,;:!?()[]")}


def score_case(case: dict) -> dict:
    expected_answer_terms = _normalize(case.get("expected_answer", ""))
    evidence_terms = _normalize(case.get("expected_evidence", ""))
    answer_supported = bool(expected_answer_terms and expected_answer_terms & evidence_terms)
    has_citations = bool(case.get("expected_citations"))
    verified = case.get("verified") is True

    faithfulness = 1.0 if verified and answer_supported and has_citations else 0.0
    answer_relevance = 1.0 if case.get("question") and case.get("expected_answer") else 0.0
    context_precision = 1.0 if case.get("expected_evidence") and has_citations else 0.0

    return {
        "id": case.get("id"),
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_precision": context_precision,
        "citation_coverage": 1.0 if has_citations else 0.0,
    }


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[dict]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_dataset(cases: list[dict]) -> dict:
    total = len(cases)
    if total == 0:
        return {
            "total_cases": 0,
            "verified_cases": 0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "answer_relevance": 0.0,
        }

    case_scores = [score_case(case) for case in cases]

    def average(metric: str) -> float:
        return sum(case[metric] for case in case_scores) / total

    verified = sum(1 for case in cases if case.get("verified") is True)

    return {
        "total_cases": total,
        "verified_cases": verified,
        "mode": "deterministic",
        "faithfulness": average("faithfulness"),
        "context_precision": average("context_precision"),
        "answer_relevance": average("answer_relevance"),
        "citation_coverage": average("citation_coverage"),
        "case_scores": case_scores,
    }


def load_quality_threshold(config_path: str | Path = DEFAULT_CONFIG) -> float:
    config_path = Path(config_path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return float(config["evaluation"]["min_faithfulness"])


def load_eval_mode(config_path: str | Path = DEFAULT_CONFIG) -> str:
    config_path = Path(config_path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return str(config.get("evaluation", {}).get("mode", DEFAULT_EVAL_MODE))


def _ragas_dataset_rows(cases: list[dict]) -> list[dict]:
    return [
        {
            "question": case["question"],
            "answer": case["expected_answer"],
            "contexts": [case["expected_evidence"]],
            "ground_truth": case["expected_answer"],
            "reference": case["expected_answer"],
        }
        for case in cases
    ]


def evaluate_dataset_with_ragas(cases: list[dict]) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        raise RuntimeError(
            "Ragas mode requires optional dependencies. Install ragas and datasets, "
            "then configure the required model credentials for your Ragas setup."
        ) from exc

    dataset = Dataset.from_list(_ragas_dataset_rows(cases))
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    scores = dict(result)
    total = len(cases)
    verified = sum(1 for case in cases if case.get("verified") is True)
    return {
        "total_cases": total,
        "verified_cases": verified,
        "mode": "ragas",
        "faithfulness": float(scores.get("faithfulness", 0.0)),
        "context_precision": float(scores.get("context_precision", 0.0)),
        "answer_relevance": float(scores.get("answer_relevancy", scores.get("answer_relevance", 0.0))),
        "citation_coverage": sum(1 for case in cases if case.get("expected_citations")) / total if total else 0.0,
        "case_scores": [],
    }


def evaluate_dataset_by_mode(cases: list[dict], mode: str = DEFAULT_EVAL_MODE) -> dict:
    mode = mode.lower()
    if mode == "deterministic":
        return evaluate_dataset(cases)
    if mode == "ragas":
        return evaluate_dataset_with_ragas(cases)
    if mode == "auto":
        try:
            return evaluate_dataset_with_ragas(cases)
        except RuntimeError:
            return evaluate_dataset(cases)
    raise ValueError(f"Unsupported evaluation mode: {mode}")


def quality_gate(metrics: dict, min_faithfulness: float) -> tuple[bool, str]:
    faithfulness = metrics["faithfulness"]
    if faithfulness >= min_faithfulness:
        return True, f"faithfulness passed: {faithfulness:.2f} >= {min_faithfulness:.2f}"
    return False, f"faithfulness failed: {faithfulness:.2f} < {min_faithfulness:.2f}"


def print_report(metrics: dict, min_faithfulness: float | None = None) -> None:
    print("RAG evaluation report")
    print(f"mode={metrics.get('mode', 'deterministic')}")
    print(f"total_cases={metrics['total_cases']}")
    print(f"verified_cases={metrics['verified_cases']}")
    print(f"faithfulness={metrics['faithfulness']:.2f}")
    if min_faithfulness is not None:
        print(f"faithfulness_threshold={min_faithfulness:.2f}")
    print(f"context_precision={metrics['context_precision']:.2f}")
    print(f"answer_relevance={metrics['answer_relevance']:.2f}")
    print(f"citation_coverage={metrics['citation_coverage']:.2f}")
    print("case_scores:")
    for case in metrics.get("case_scores", []):
        print(
            "- "
            f"id={case['id']} "
            f"faithfulness={case['faithfulness']:.2f} "
            f"context_precision={case['context_precision']:.2f} "
            f"answer_relevance={case['answer_relevance']:.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation over golden examples.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--min-faithfulness", type=float, default=None)
    parser.add_argument("--mode", choices=["deterministic", "ragas", "auto"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    min_faithfulness = (
        args.min_faithfulness
        if args.min_faithfulness is not None
        else load_quality_threshold(args.config)
    )
    mode = args.mode or load_eval_mode(args.config)
    metrics = evaluate_dataset_by_mode(load_dataset(args.dataset), mode=mode)
    print_report(metrics, min_faithfulness=min_faithfulness)
    passed, message = quality_gate(metrics, min_faithfulness)
    print(message)
    if not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
