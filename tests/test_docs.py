from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_quickstart_documents_fresh_clone_flow():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "python3 -m venv .venv" in readme
    assert "python -m pip install -r requirements.txt" in readme
    assert "python scripts/ingest.py --pdf docs.pdf --build-vector-db" in readme
    assert "python -m uvicorn main:app --reload" in readme
    assert "chroma_db/" in readme
    assert "[PRD](docs/PRD.md)" in readme
    assert "[Architecture](docs/ARCHITECTURE.md)" in readme
    assert "[Testing Strategy](docs/TESTING.md)" in readme


def test_testing_strategy_documents_baseline_test_command():
    testing = (ROOT / "docs" / "TESTING.md").read_text(encoding="utf-8")

    assert "python -m pytest" in testing
    assert "Chunking" in testing
    assert "Chroma persistence" in testing
    assert ".github/workflows/rag-eval.yml" in testing
    assert "prompt hardening" in testing.lower()


def test_readme_documents_runtime_configuration():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert ".env.example" in readme
    assert "src.rag.config.load_settings" in readme
    assert "chunk size" in readme
    assert "optional LLM/reranker provider" in readme


def test_readme_contains_portfolio_demo_walkthrough():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Portfolio Demo" in readme
    assert "python scripts/query.py \"What is a GitHub Actions runner?\"" in readme
    assert '"retrieval_mode":"hybrid"' in readme
    assert "The answer is not available in the retrieved context." in readme
    assert "faithfulness passed: 1.00 >= 0.90" in readme
    assert "[Reranking](docs/RERANKING.md)" in readme


def test_prd_documents_product_phases_and_metrics():
    prd = (ROOT / "docs" / "PRD.md").read_text(encoding="utf-8")

    assert "Product Goal" in prd
    assert "Non-Goals" in prd
    assert "Phase 1" in prd
    assert "Phase 4" in prd
    assert "Success Metrics" in prd
    assert "issues/1" in prd
    assert "issues/20" in prd


def test_architecture_documents_data_flow_and_interfaces():
    architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "Data Flow" in architecture
    assert "src/rag/chunking.py" in architecture
    assert "POST /query" in architecture
    assert "chroma_db/" in architecture
    assert "python evals/run_ragas.py --config configs/settings.toml" in architecture


def test_pr_workflow_and_template_document_issue_tracking():
    workflow = (ROOT / "docs" / "PR_WORKFLOW.md").read_text(encoding="utf-8")
    template = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "codex/issue-<number>-<short-description>" in workflow
    assert "one branch for one issue" in workflow
    assert "Closes #<issue-number>" in workflow
    assert "Linked Issue" in template
    assert "Validation" in template
    assert "Screenshots Or Logs" in template
    assert "[Pull Request Workflow](docs/PR_WORKFLOW.md)" in readme


def test_backup_recovery_documents_generated_state_and_rebuild():
    recovery = (ROOT / "docs" / "BACKUP_RECOVERY.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "chroma_db/" in recovery
    assert "data/processed/ingestion_manifest.json" in recovery
    assert "python scripts/ingest.py --pdf docs.pdf --build-vector-db" in recovery
    assert "evals/golden.jsonl" in recovery
    assert ".env" in recovery
    assert "[Backup And Recovery](docs/BACKUP_RECOVERY.md)" in readme


def test_drift_metrics_doc_defines_signals_and_thresholds():
    drift = (ROOT / "docs" / "DRIFT_METRICS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Query distribution" in drift
    assert "No-answer rate" in drift
    assert "citation_coverage_min" in drift
    assert "configs/settings.toml" in drift
    assert "[Drift Metrics](docs/DRIFT_METRICS.md)" in readme
