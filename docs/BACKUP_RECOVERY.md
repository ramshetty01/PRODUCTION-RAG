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

If generated state is deleted, rebuild it from the source documents.

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

## Recovery Checks

After rebuilding:

```bash
python scripts/query.py "What is a GitHub Actions workflow?"
python -m pytest
python evals/run_ragas.py --config configs/settings.toml
```

## Backup Notes

- Keep source documents in durable storage before relying on generated vectors.
- Back up `evals/golden.jsonl` whenever production feedback is promoted into
  evaluation cases.
- Do not back up `.env` into Git or shared artifact storage if it contains
  secrets.
- Rebuild generated vector state after dependency, prompt, chunking, or source
  document changes.
