from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import DEFAULT_DB_PATH, DEFAULT_PDF_PATH, chunk_pdf, chunk_text_file, chunk_token_summary
from src.rag.config import load_settings
from src.rag.ingestion import DEFAULT_MANIFEST, load_manifest, plan_document_ingestion, record_document_ingestion, save_manifest
from src.rag.vector_store import build_chroma_db, count_records


DEFAULT_CORPUS = [
    PROJECT_ROOT / "data" / "raw" / "enterprise-security-handbook.md",
    PROJECT_ROOT / "data" / "raw" / "vendor-risk-policy.md",
]
SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest the demo enterprise corpus into ChromaDB.")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Source document to ingest. Repeat for multiple files.",
    )
    parser.add_argument(
        "--include-sample-pdf",
        action="store_true",
        help="Also ingest the original docs.pdf sample.",
    )
    parser.add_argument("--persist-dir", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    return parser.parse_args()


def _source_paths(args) -> list[Path]:
    paths = [Path(source) for source in (args.sources or DEFAULT_CORPUS)]
    if args.include_sample_pdf:
        paths.append(DEFAULT_PDF_PATH)
    return paths


def _chunk_source(path: Path, document_version: str, chunk_size: int, chunk_overlap: int):
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported source type: {path}")
    if suffix == ".pdf":
        return chunk_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, document_version=document_version)
    return chunk_text_file(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, document_version=document_version)


def ingest_sources(
    sources: list[Path],
    persist_dir: str | Path = DEFAULT_DB_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    settings = load_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    manifest = load_manifest(manifest_path)
    indexed = []
    skipped = []

    for source in sources:
        source = source.expanduser().resolve()
        decision = plan_document_ingestion(source, manifest)
        if not decision.should_reindex:
            skipped.append({"source": str(source), "document_id": decision.document_id})
            continue

        chunks = _chunk_source(source, decision.document_version, chunk_size, chunk_overlap)
        vectorstore = build_chroma_db(chunks, persist_directory=Path(persist_dir))
        record_document_ingestion(manifest, decision, source, chunk_count=len(chunks))
        indexed.append(
            {
                "source": str(source),
                "document_id": decision.document_id,
                "document_version": decision.document_version,
                "chunks": len(chunks),
                "chunk_summary": chunk_token_summary(chunks),
                "vector_records": count_records(vectorstore),
            }
        )

    save_manifest(manifest, manifest_path)
    return {"indexed": indexed, "skipped": skipped}


def main() -> int:
    args = parse_args()
    result = ingest_sources(
        _source_paths(args),
        persist_dir=args.persist_dir,
        manifest_path=args.manifest,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    for item in result["indexed"]:
        print(
            f"Indexed {Path(item['source']).name}: "
            f"{item['document_id']}@{item['document_version']} "
            f"chunks={item['chunks']} records={item['vector_records']}"
        )
    for item in result["skipped"]:
        print(f"Skipped {Path(item['source']).name}: unchanged {item['document_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
