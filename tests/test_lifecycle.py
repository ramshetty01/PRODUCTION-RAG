from src.rag.ingestion import (
    load_manifest,
    plan_document_ingestion,
    record_document_ingestion,
    save_manifest,
)


def test_ingestion_manifest_skips_unchanged_documents(tmp_path):
    source = tmp_path / "docs.pdf"
    source.write_text("version one", encoding="utf-8")
    manifest = {"documents": {}}

    first = plan_document_ingestion(source, manifest)
    record_document_ingestion(manifest, first, source, chunk_count=2)
    second = plan_document_ingestion(source, manifest)

    assert first.should_reindex is True
    assert first.document_version == "v1"
    assert second.should_reindex is False
    assert second.reason == "unchanged"
    assert second.document_version == "v1"


def test_ingestion_manifest_versions_changed_documents(tmp_path):
    source = tmp_path / "docs.pdf"
    source.write_text("version one", encoding="utf-8")
    manifest = {"documents": {}}
    first = plan_document_ingestion(source, manifest)
    record_document_ingestion(manifest, first, source, chunk_count=2)

    source.write_text("version two", encoding="utf-8")
    changed = plan_document_ingestion(source, manifest)
    record_document_ingestion(manifest, changed, source, chunk_count=3)

    record = manifest["documents"]["docs"]
    assert changed.should_reindex is True
    assert changed.reason == "changed"
    assert changed.document_version == "v2"
    assert record["previous_version"] == "v1"
    assert record["chunk_count"] == 3


def test_ingestion_manifest_round_trips_to_disk(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = {"documents": {"docs": {"document_version": "v1"}}}

    save_manifest(manifest, manifest_path)

    assert load_manifest(manifest_path) == manifest
