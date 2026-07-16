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
- `category`: coverage bucket such as factual, lexical, citation-heavy,
  refusal, or multi-hop.
- `permission_level`: public or restricted coverage slice.
- `retrieval_mode`: expected retrieval slice such as semantic, exact, hybrid,
  sparse, or reranked.

The committed dataset contains verified examples across `docs.pdf`, the
`evals/fixtures/security-policy.md` representative policy fixture, and the
enterprise demo corpus under `data/raw/`. It covers factual definition
questions, exact lexical queries, citation-heavy checks, refusal boundaries,
multi-hop questions, permission-sensitive examples, adversarial prompt attempts,
vendor-risk policy, security operations policy, and retrieval-mode slices for
semantic, exact, hybrid, sparse, and reranked behavior. Grow it toward 100-200
verified examples as more source documents and production failure cases are
added.

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

## Evaluation Modes

The default mode is deterministic so CI does not require model credentials:

```bash
python evals/run_ragas.py --config configs/settings.toml --mode deterministic
```

To run full Ragas metrics, install optional dependencies and configure the LLM
and embedding credentials required by your Ragas setup:

```bash
python -m pip install ragas datasets
python evals/run_ragas.py --config configs/settings.toml --mode ragas
```

`--mode auto` attempts the Ragas path first and falls back to deterministic
scoring if optional dependencies are unavailable.
