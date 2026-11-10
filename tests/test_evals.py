import json
from pathlib import Path

from evals.run_ragas import (
    evaluate_dataset,
    evaluate_dataset_by_mode,
    load_dataset,
    load_eval_mode,
    load_quality_threshold,
    quality_gate,
    score_case,
)


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
    "category",
    "permission_level",
    "retrieval_mode",
}


def load_cases():
    return [
        json.loads(line)
        for line in GOLDEN_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_golden_dataset_has_initial_verified_examples():
    cases = load_cases()

    assert len(cases) >= 66
    assert all(case["verified"] is True for case in cases)
    assert {"factual", "lexical", "citation-heavy", "refusal", "multi-hop", "permission-sensitive", "adversarial"} <= {
        case["category"] for case in cases
    }
    assert {"public", "restricted"} <= {case["permission_level"] for case in cases}
    assert {"semantic", "exact", "hybrid", "sparse", "reranked"} <= {case["retrieval_mode"] for case in cases}
    assert len({case["source"] for case in cases}) >= 2
    assert {"enterprise-security-handbook.md", "vendor-risk-policy.md"} <= {case["source"] for case in cases}


def test_golden_dataset_rows_have_required_schema():
    for case in load_cases():
        assert REQUIRED_FIELDS <= case.keys()
        assert case["id"]
        assert case["question"].endswith("?")
        assert case["expected_answer"]
        assert case["expected_evidence"]
        assert case["source"]
        assert isinstance(case["page"], int)
        assert isinstance(case["expected_citations"], list)
        assert case["expected_citations"]
        assert case["permission_level"] in {"public", "restricted"}
        assert case["retrieval_mode"] in {"semantic", "exact", "hybrid", "sparse", "reranked"}


def test_reference_only_validation_does_not_claim_model_quality():
    metrics = evaluate_dataset(load_dataset(GOLDEN_DATASET))
    assert metrics["total_cases"] >= 66
    assert metrics["status"] == "not_run"
    assert metrics["faithfulness"] is None
    assert metrics["citation_coverage"] is None
    assert metrics["refusal_accuracy"] is None
    assert metrics["latency_budget_pass"] is None
    assert metrics["case_scores"] == []
    assert not quality_gate(metrics, 0.9)[0]


def test_diagnostics_use_actual_predictions_and_keep_missing_rows():
    cases = load_dataset(GOLDEN_DATASET)[:2]
    predictions = [{"id": cases[0]["id"], "answer": "invented unsupported words", "contexts": ["actual retrieved passage"], "citations": []}]
    metrics = evaluate_dataset(cases, predictions)
    assert metrics["missing_prediction_cases"] == 1
    assert metrics["lexical_support_proxy"] == 0
    assert metrics["reference_token_recall"] == 0
    assert metrics["faithfulness"] is None
    assert metrics["citation_coverage"] == 0
    assert len(metrics["case_scores"]) == 2
    supported = {**predictions[0], "answer": "actual retrieved passage", "citations": ["source-1"]}
    assert score_case(cases[0], supported)["lexical_support_proxy"] == 1
    assert evaluate_dataset(cases, [supported])["citation_coverage"] == 0.5


def test_ragas_never_receives_gold_as_actual_prediction():
    from evals.run_ragas import _ragas_dataset_rows
    cases = load_dataset(GOLDEN_DATASET)[:1]
    predictions = [{"id": cases[0]["id"], "answer": "actual answer", "contexts": ["actual evidence"], "citations": []}]
    rows = _ragas_dataset_rows(cases, predictions)
    assert rows[0]["answer"] == "actual answer"
    assert rows[0]["contexts"] == ["actual evidence"]
    assert rows[0]["reference"] == cases[0]["expected_answer"]


def test_invalid_or_duplicate_predictions_fail_closed():
    import pytest
    from evals.run_ragas import _ragas_dataset_rows
    cases = load_dataset(GOLDEN_DATASET)[:1]
    row = {"id": cases[0]["id"], "answer": "answer", "contexts": [], "citations": []}
    for predictions in ([row, row], [{**row, "id": "unknown"}], [{**row, "contexts": "not a list"}], [{**row, "retrieval_ms": float("nan")} ]):
        with pytest.raises(ValueError):
            evaluate_dataset(cases, predictions)
    with pytest.raises(ValueError):
        _ragas_dataset_rows(cases, [])
    with pytest.raises(ValueError):
        evaluate_dataset_by_mode(cases, "ragas")
    assert not quality_gate({"faithfulness": float("nan")}, .9)[0]


def test_quality_gate_passes_or_fails_against_threshold():
    passing, pass_message = quality_gate({"faithfulness": 0.90}, min_faithfulness=0.90)
    failing, fail_message = quality_gate({"faithfulness": 0.89}, min_faithfulness=0.90)

    assert passing is True
    assert pass_message == "faithfulness passed: 0.90 >= 0.90"
    assert failing is False
    assert fail_message == "faithfulness failed: 0.89 < 0.90"


def test_quality_threshold_loads_from_versioned_config(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text("[evaluation]\nmin_faithfulness = 0.91\nmode = \"auto\"\n", encoding="utf-8")

    assert load_quality_threshold(config) == 0.91
    assert load_eval_mode(config) == "auto"


def test_evaluate_dataset_by_mode_preserves_deterministic_default():
    metrics = evaluate_dataset_by_mode(load_dataset(GOLDEN_DATASET), mode="deterministic")

    assert metrics["mode"] == "deterministic"
    assert metrics["faithfulness"] is None


def test_evaluate_dataset_by_mode_rejects_unknown_mode():
    try:
        evaluate_dataset_by_mode([], mode="unknown")
    except ValueError as exc:
        assert "Unsupported evaluation mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_cli_fixture_success_cannot_pass_required_quality_gate():
    import subprocess
    import sys

    root = GOLDEN_DATASET.parents[1]
    command = [sys.executable, "evals/run_ragas.py", "--config", "configs/settings.toml"]
    fixture = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert fixture.returncode == 0
    assert '"faithfulness": null' in fixture.stdout
    assert "Fixture validation/diagnostics only" in fixture.stdout
    required = subprocess.run(command + ["--require-quality-gate"], cwd=root, capture_output=True, text=True)
    assert required.returncode == 1
    assert "quality gate not evaluated" in required.stdout


def test_prediction_latency_and_refusal_require_actual_measurements():
    case = {**load_dataset(GOLDEN_DATASET)[0], "category": "refusal"}
    row = {"id": case["id"], "answer": "Cannot answer", "contexts": [], "citations": [], "refused": True, "retrieval_ms": 2500}
    result = evaluate_dataset([case], [row])
    assert result["refusal_accuracy"] == 1
    assert result["latency_budget_pass"] is False
    row.pop("retrieval_ms")
    assert evaluate_dataset([case], [row])["latency_budget_pass"] is None
    assert not quality_gate({"faithfulness": 2}, .9)[0]
