from pathlib import Path

from scripts.ingest_corpus import DEFAULT_CORPUS, ingest_sources


ROOT = Path(__file__).resolve().parents[1]


def test_enterprise_demo_corpus_documents_are_committed():
    sources = {path.name for path in DEFAULT_CORPUS}

    assert {"enterprise-security-handbook.md", "vendor-risk-policy.md"} <= sources
    for source in DEFAULT_CORPUS:
        text = source.read_text(encoding="utf-8")
        assert "retrieved context" in text or "SOC 2 Type II" in text


def test_ingest_corpus_indexes_markdown_sources(tmp_path, monkeypatch):
    source = tmp_path / "vendor-risk-policy.md"
    source.write_text("Vendors must provide SOC 2 Type II evidence before onboarding.", encoding="utf-8")
    built = {}

    def fake_build_vector_db(chunks, persist_directory, settings):
        built["chunks"] = chunks
        built["persist_directory"] = persist_directory
        built["backend"] = settings.vector_backend
        return object()

    monkeypatch.setattr("scripts.ingest_corpus.build_vector_db", fake_build_vector_db)
    monkeypatch.setattr("scripts.ingest_corpus.count_records", lambda _vectorstore: len(built["chunks"]))

    result = ingest_sources(
        [source],
        persist_dir=tmp_path / "chroma_db",
        manifest_path=tmp_path / "manifest.json",
        chunk_size=20,
        chunk_overlap=2,
    )

    assert result["skipped"] == []
    assert result["indexed"][0]["document_id"] == "vendor-risk-policy"
    assert result["indexed"][0]["chunks"] == len(built["chunks"])
    assert built["persist_directory"] == tmp_path / "chroma_db"
    assert built["backend"] == "chroma"
    assert (tmp_path / "manifest.json").exists()
