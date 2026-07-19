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

## Load And Reliability Testing

Run the local API with an indexed corpus, then execute the lightweight load
test:

```bash
python -m uvicorn main:app --reload
python scripts/load_test.py http://localhost:8000 \
  --profile standard \
  --requests-per-endpoint 25 \
  --concurrency 8 \
  --api-key public-key \
  --output reports/load-test.json
```

The standard profile exercises `/health`, `/metrics`, `/upload`,
`/index-status`, `/query`, and `/query/stream`. The JSON report includes the
profile, concurrency, request count, status counts, error rate, rate-limited
request count, p50/p95/max latency, and query cache hit rate.

Use the CI-safe smoke profile when you only need endpoint coverage:

```bash
python scripts/load_test.py http://localhost:8000 --profile smoke --api-key public-key
```

Run abuse pressure separately so intentional failures do not mask launch
latency:

```bash
python scripts/load_test.py http://localhost:8000 \
  --profile abuse \
  --requests-per-endpoint 5 \
  --concurrency 12 \
  --api-key public-key \
  --output reports/abuse-test.json
```

Expected local thresholds:

- p95 latency at or below 3000 ms for the combined endpoint mix.
- error rate at or below 1 percent, excluding intentional 429 rate-limit
  pressure tests.
- 429 responses should appear when concurrency intentionally exceeds the
  configured request window.
- oversized uploads should return a 4xx response without creating chunks.
- concurrent background upload jobs should still leave `/index-status`
  responsive.
- streaming requests should complete without 5xx responses under the standard
  profile.
- cache hit rate should increase when repeated `/query` payloads are used.

## Adding Tests

When adding behavior, include a focused test near the behavior owner:

- `tests/test_chunking.py` for chunk generation and metadata.
- `tests/test_ingestion.py` and `tests/test_lifecycle.py` for indexing state.
- `tests/test_retrieval.py` for retrieval, generation, ranking, and refusal.
- `tests/test_api.py` for API contract behavior.
- `tests/test_evals.py` for evaluation dataset and quality gates.
- `tests/test_load_testing.py` for load-test report calculations.
