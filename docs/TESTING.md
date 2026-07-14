# Testing Strategy

The project uses `pytest` for local and CI validation.

## Local Command

```bash
python -m pytest
```

## Baseline Coverage

Current baseline tests cover:

- Chunking: token target, overlap, and citation metadata.
- Ingestion and vector storage: Chroma persistence and reload behavior.
- Retrieval: vector retrieval, metadata filters, permissions, hybrid search, and
  reranking.
- Citation grounding and refusal behavior.
- Prompt hardening: delimiter use, instruction hierarchy, fake citations,
  unsupported claims, and adversarial prompt attempts.
- API behavior: health, query, feedback, monitoring, and structured errors.
- Evaluation: golden dataset schema, offline scoring, and quality gate.
- Operations: lifecycle manifest, deployment artifacts, observability,
  performance controls, feedback monitoring, and security checks.

## Test Data

Tests use small in-memory documents and fake embedding/vector-store objects where
possible. This keeps local runs fast and avoids external model downloads for unit
tests.

## CI

Pull requests run the full `python -m pytest` suite through
`.github/workflows/rag-eval.yml` and then execute the offline RAG evaluation
gate.

Dependency or container changes also run `.github/workflows/security-scan.yml`.
That workflow audits Python dependencies with `pip-audit` and scans the
repository, including the Dockerfile, with Trivy. The workflow uploads JSON
reports as artifacts so the current security baseline is visible during
dependency and container changes.

Run the same scans locally before changing production dependencies or container
settings:

```bash
python -m pip install pip-audit==2.7.3
pip-audit -r requirements.txt --progress-spinner off
trivy fs .
```

## Adding Tests

When adding behavior, include a focused test near the behavior owner:

- `tests/test_chunking.py` for chunk generation and metadata.
- `tests/test_ingestion.py` and `tests/test_lifecycle.py` for indexing state.
- `tests/test_retrieval.py` for retrieval, generation, ranking, and refusal.
- `tests/test_api.py` for API contract behavior.
- `tests/test_evals.py` for evaluation dataset and quality gates.
