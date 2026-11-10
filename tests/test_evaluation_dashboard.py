import json

from src.rag.evaluation_report import build_evaluation_report


def test_build_evaluation_report_exposes_reviewer_metrics():
    report = build_evaluation_report()

    assert report["dataset"]["total_cases"] >= 66
    assert report["dataset"]["verified_cases"] == report["dataset"]["total_cases"]
    assert report["metrics"]["faithfulness"] is None
    assert report["metrics"]["citation_coverage"] is None
    assert report["metrics"]["refusal_accuracy"] is None
    assert report["metrics"]["latency_budget_pass"] is None
    assert report["quality_gate"]["passed"] is None
    assert "not measured" in report["quality_gate"]["message"]
    assert report["case_scores"] == []


def test_export_evaluation_report_writes_dashboard_json(tmp_path):
    from scripts.export_evaluation_report import main

    output = tmp_path / "evaluation-dashboard.json"
    import sys

    original_argv = sys.argv
    try:
        sys.argv = ["export_evaluation_report.py", "--output", str(output)]
        assert main() == 0
    finally:
        sys.argv = original_argv

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["quality_gate"]["passed"] is None
    assert report["metrics"]["refusal_accuracy"] is None
