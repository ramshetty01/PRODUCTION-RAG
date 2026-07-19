from __future__ import annotations

import argparse
import json
from pathlib import Path


def readable_json(path: Path) -> bool:
    if not path.exists():
        return False
    json.loads(path.read_text(encoding="utf-8"))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify restored RAG runtime state.")
    parser.add_argument("--manifest", default="data/processed/ingestion_manifest.json")
    parser.add_argument("--vector-db", default="chroma_db")
    parser.add_argument("--uploads", default="data/uploads")
    parser.add_argument("--logs", default="logs")
    args = parser.parse_args()

    checks = {
        "manifest": readable_json(Path(args.manifest)),
        "vector_db": Path(args.vector_db).exists(),
        "uploads": Path(args.uploads).exists(),
        "logs": Path(args.logs).exists(),
    }
    for name, ok in checks.items():
        print(f"{name}: {'ok' if ok else 'missing'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
