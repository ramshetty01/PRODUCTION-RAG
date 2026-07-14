import json
from pathlib import Path

from evals.run_ragas import evaluate_dataset, load_dataset, load_quality_threshold, quality_gate


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


def test_quality_gate_passes_or_fails_against_threshold():
    passing, pass_message = quality_gate({"faithfulness": 0.90}, min_faithfulness=0.90)
    failing, fail_message = quality_gate({"faithfulness": 0.89}, min_faithfulness=0.90)

    assert passing is True
    assert pass_message == "faithfulness passed: 0.90 >= 0.90"
    assert failing is False
    assert fail_message == "faithfulness failed: 0.89 < 0.90"


def test_quality_threshold_loads_from_versioned_config(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text("[evaluation]\nmin_faithfulness = 0.91\n", encoding="utf-8")

    assert load_quality_threshold(config) == 0.91
