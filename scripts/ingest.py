from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_DB_PATH,
    DEFAULT_PDF_PATH,
    chunk_pdf,
    chunk_token_summary,
    count_tokens,
)
from src.rag.config import load_settings
from src.rag.ingestion import (
    DEFAULT_MANIFEST,
    load_manifest,
    plan_document_ingestion,
    record_document_ingestion,
    save_manifest,
)
from src.rag.vector_store import build_vector_db, count_records


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest a PDF into retrieval-ready chunks.")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF_PATH), help="Path to the source PDF.")
    parser.add_argument(
        "--persist-dir",
        default=str(DEFAULT_DB_PATH),
        help="Directory where ChromaDB should be written.",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--build-vector-db",
        action="store_true",
        help="Also build ChromaDB after splitting. This belongs to the vector storage step.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings()
    manifest = load_manifest(args.manifest)
    decision = plan_document_ingestion(args.pdf, manifest)
    if not decision.should_reindex:
        print(f"Skipped {Path(args.pdf).name}: unchanged document {decision.document_id}@{decision.document_version}")
        return

    chunks = chunk_pdf(
        args.pdf,
        chunk_size=args.chunk_size or settings.chunk_size,
        chunk_overlap=args.chunk_overlap or settings.chunk_overlap,
        document_version=decision.document_version,
    )
    token_counts = [count_tokens(chunk.page_content) for chunk in chunks]
    summary = chunk_token_summary(chunks)

    print(f"Loaded {Path(args.pdf).name}")
    print(f"Created {len(chunks)} chunks")
    if token_counts:
        print(
            "Token counts: "
            f"min={min(token_counts)}, max={max(token_counts)}, "
            f"target={args.chunk_size or settings.chunk_size}, "
            f"overlap={args.chunk_overlap or settings.chunk_overlap}"
        )
        print(f"Chunk verification: {summary}")

    if args.build_vector_db:
        vectorstore = build_vector_db(chunks, persist_directory=Path(args.persist_dir), settings=settings)
        print(f"Saved Chroma database to {args.persist_dir}")
        print(f"Chroma records: {count_records(vectorstore)}")

    record_document_ingestion(manifest, decision, args.pdf, chunk_count=len(chunks))
    save_manifest(manifest, args.manifest)
    print(f"Recorded {decision.document_id}@{decision.document_version} in {args.manifest}")


if __name__ == "__main__":
    main()
