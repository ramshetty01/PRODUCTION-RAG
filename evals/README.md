# Golden Evaluation Dataset

`golden.jsonl` contains manually verified examples for offline RAG evaluation.
Each line is one JSON object with these fields:

- `id`: stable case identifier.
- `question`: user question to run through the RAG pipeline.
- `expected_answer`: concise answer verified against source evidence.
- `expected_citations`: chunk IDs that should support the answer.
- `expected_evidence`: source text or supporting paragraph.
- `source`: source document name.
- `page`: zero-based PDF page number from the loader metadata.
- `verified`: must be `true` for cases used in quality gates.

The initial dataset is intentionally small. Grow it toward 50-200 verified
examples as more source documents and failure cases are added.

## Running Offline Evaluation

Run:

```bash
python evals/run_ragas.py --config configs/settings.toml
```

The script reads `golden.jsonl`, scores each case, prints faithfulness as the
primary metric, and also reports context precision, answer relevance, and
citation coverage. The current implementation is deterministic and dependency
light so it can run in CI; the metric names and report shape are compatible with
a later full Ragas-backed evaluator.
