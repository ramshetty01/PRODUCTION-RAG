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


def test_readme_documents_runtime_configuration():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert ".env.example" in readme
    assert "src.rag.config.load_settings" in readme
    assert "chunk size" in readme
    assert "optional LLM provider" in readme


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
