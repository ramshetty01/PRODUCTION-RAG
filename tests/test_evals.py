import json
from pathlib import Path

from evals.run_ragas import evaluate_dataset, load_dataset


GOLDEN_DATASET = Path(__file__).resolve().parents[1] / "evals" / "golden.jsonl"


REQUIRED_FIELDS = {
    "id",
    "question",
    "expected_answer",
    "expected_citations",
    "expected_evidence",
    "source",
    "page",
    "verified",
}


def load_cases():
    return [
        json.loads(line)
        for line in GOLDEN_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_golden_dataset_has_initial_verified_examples():
    cases = load_cases()

    assert len(cases) >= 5
    assert all(case["verified"] is True for case in cases)


def test_golden_dataset_rows_have_required_schema():
    for case in load_cases():
        assert REQUIRED_FIELDS <= case.keys()
        assert case["id"]
        assert case["question"].endswith("?")
        assert case["expected_answer"]
        assert case["expected_evidence"]
        assert case["source"] == "docs.pdf"
        assert isinstance(case["page"], int)
        assert isinstance(case["expected_citations"], list)
        assert case["expected_citations"]


def test_offline_eval_reports_quality_metrics():
    metrics = evaluate_dataset(load_dataset(GOLDEN_DATASET))

    assert metrics["total_cases"] >= 5
    assert metrics["faithfulness"] == 1.0
    assert metrics["context_precision"] == 1.0
    assert metrics["answer_relevance"] == 1.0
    assert metrics["citation_coverage"] == 1.0
