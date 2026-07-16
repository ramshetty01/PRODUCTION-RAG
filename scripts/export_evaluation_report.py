from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.run_ragas import DEFAULT_CONFIG, DEFAULT_DATASET
from src.rag.evaluation_report import build_evaluation_report


def parse_args():
    parser = argparse.ArgumentParser(description="Export the reviewer evaluation dashboard report as JSON.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", choices=["deterministic", "ragas", "auto"], default=None)
    parser.add_argument("--output", default="reports/evaluation-dashboard.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_evaluation_report(args.dataset, args.config, mode=args.mode)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote evaluation dashboard report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
