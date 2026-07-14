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
