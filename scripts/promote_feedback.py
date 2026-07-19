from __future__ import annotations

import argparse

from src.rag.monitoring import DEFAULT_FEEDBACK_LOG, load_feedback, write_draft_eval_cases


def parse_args():
    parser = argparse.ArgumentParser(description="Convert production feedback into draft eval cases for human review.")
    parser.add_argument("--feedback", default=str(DEFAULT_FEEDBACK_LOG))
    parser.add_argument("--output", default="evals/drafts/feedback-candidates.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = write_draft_eval_cases(load_feedback(args.feedback), args.output)
    print(f"Wrote {count} draft eval cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
