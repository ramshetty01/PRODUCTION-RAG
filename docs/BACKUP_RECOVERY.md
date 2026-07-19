# Backup And Recovery

## Source Of Truth

Back up these committed or source-managed files:

- Source documents: `docs.pdf` and future files under `data/raw/`.
- Evaluation datasets: `evals/golden.jsonl` and supporting eval docs.
- Runtime configuration templates: `.env.example` and `configs/settings.toml`.
- Prompt files: `prompts/*.md`.
- Source code and tests.

## Generated State

These files are generated and should not be committed:

- `chroma_db/`: local Chroma vector database.
- `data/processed/ingestion_manifest.json`: document lifecycle manifest.
- `logs/*.jsonl`: runtime logs and feedback.
- `data/uploads/`: uploaded source files.

In production, back up the manifest, vector DB, uploaded files, and logs
together so document metadata, searchable vectors, source evidence, audit logs,
feedback, and deletion proof stay consistent.

## Backup

Create an artifact from the project root:

```bash
tar -czf backups/rag-state-$(date +%Y%m%d%H%M%S).tgz \
  data/processed/ingestion_manifest.json \
  data/uploads \
  chroma_db \
  logs
```

Store the archive in encrypted object storage with retention and access logs.
Do not include `.env`.

## Restore

Stop writes, restore the archive, then run the smoke check:

```bash
tar -xzf backups/rag-state-YYYYMMDDHHMMSS.tgz
python scripts/restore_smoke.py
python scripts/query.py "What is a GitHub Actions workflow?"
```

For managed vector databases, restore the provider snapshot first, then restore
`data/processed/ingestion_manifest.json`, `data/uploads`, and `logs`.

## Rebuild ChromaDB

From a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/ingest.py --pdf docs.pdf --build-vector-db
```

This recreates:

- `chroma_db/`
- `data/processed/ingestion_manifest.json`

Use rebuild only when source documents are complete and a point-in-time vector
snapshot is unavailable.

## Recovery Checks

After rebuilding:

```bash
python scripts/restore_smoke.py
python scripts/query.py "What is a GitHub Actions workflow?"
python -m pytest
python evals/run_ragas.py --config configs/settings.toml
```

## Deletion Proof

Admin delete, workspace purge, and scheduled retention write audit events under
`logs/audit.jsonl`. Workspace purge records `documents_deleted`,
`files_deleted`, `vector_records_deleted`, `conversations_deleted`, and
`logs_deleted` so operators can prove what was removed.

## Backup Notes

- Keep source documents in durable storage before relying on generated vectors.
- Back up `evals/golden.jsonl` whenever production feedback is promoted into
  evaluation cases.
- Do not back up `.env` into Git or shared artifact storage if it contains
  secrets.
- Rebuild generated vector state after dependency, prompt, chunking, or source
  document changes.
