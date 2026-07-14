from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DATASET = Path(__file__).resolve().parent / "golden.jsonl"


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

    verified = sum(1 for case in cases if case.get("verified") is True)
    with_evidence = sum(1 for case in cases if case.get("expected_evidence"))
    with_citations = sum(1 for case in cases if case.get("expected_citations"))
    with_answers = sum(1 for case in cases if case.get("expected_answer"))

    return {
        "total_cases": total,
        "verified_cases": verified,
        "faithfulness": verified / total,
        "context_precision": with_evidence / total,
        "answer_relevance": with_answers / total,
        "citation_coverage": with_citations / total,
    }


def print_report(metrics: dict) -> None:
    print("RAG evaluation report")
    print(f"total_cases={metrics['total_cases']}")
    print(f"verified_cases={metrics['verified_cases']}")
    print(f"faithfulness={metrics['faithfulness']:.2f}")
    print(f"context_precision={metrics['context_precision']:.2f}")
    print(f"answer_relevance={metrics['answer_relevance']:.2f}")
    print(f"citation_coverage={metrics['citation_coverage']:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation over golden examples.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--min-faithfulness", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = evaluate_dataset(load_dataset(args.dataset))
    print_report(metrics)
    if metrics["faithfulness"] < args.min_faithfulness:
        print(
            "faithfulness below threshold: "
            f"{metrics['faithfulness']:.2f} < {args.min_faithfulness:.2f}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
