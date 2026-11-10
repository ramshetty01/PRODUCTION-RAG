# Evaluation provenance and actual predictions

The historical evaluator fed expected answers and expected evidence into its scorer,
including optional Ragas. The old displayed faithfulness=1.0 was a reference-fixture
self-check, not a measurement of current retrieval/generation. It must not be used as
a resume claim, production gate or historical model-quality result.

## Default offline command

`python evals/run_ragas.py --config configs/settings.toml` validates reference IDs and
required strings. With no prediction file it returns null semantic metrics, status
`not_run` and no passing quality gate. Its successful exit code means fixture validation
only. The inherited `verified` flags do not establish who reviewed the references or
whether the data is held out.

## Explicit prediction contract

Capture actual outputs separately from reference labels. Each JSONL record is:

```json
{"id":"001-actions-purpose","answer":"Actual generated answer","contexts":["Actual retrieved chunk text"],"citations":["actual-chunk-id"],"refused":false,"retrieval_ms":12.3}
```

Required: id, answer, contexts, citations. Optional: a boolean refused decision and
measured retrieval_ms. IDs join reference cases. Unknown/duplicate IDs, invalid field
types and non-finite latency are rejected. Missing rows remain in deterministic metric
denominators. Do not populate answer/contexts from expected_answer/expected_evidence.
Record the model, provider, prompt/code revision, corpus snapshot and collection time
alongside the prediction artifact before publishing comparisons.

```sh
python evals/run_ragas.py --predictions predictions.jsonl --output report.json
python scripts/export_evaluation_report.py --predictions predictions.jsonl --output reports/evaluation-dashboard.json
```

Offline diagnostics are lexical-support overlap, reference-token recall and citation
presence. They do not measure semantic entailment, factual correctness or citation
validity. Refusal accuracy is available only with supplied predictions and refusal
reference cases; missing predictions count as failures. Latency passes only when every
case supplies an actual latency and all meet the 2,000 ms budget. Neither the budget
itself nor reference answers count as a measurement.

## Optional semantic judge

`--mode ragas --predictions predictions.jsonl --require-quality-gate` requires a complete
prediction set, optional Ragas dependencies and configured judge credentials. This is
explicit provider usage and may cost money. Judge failures/invalid metrics fail closed;
there is no silent passing deterministic fallback. `auto` now means offline diagnostics
and never starts a paid judge. This maintenance run did not validate a live Ragas judge,
its deployment-specific model configuration or the current optional dependency stack.

The dashboard shows Not evaluated/Not measured without actual semantic judgments.
Use independent human grading and fresh held-out sources before claiming real-world
faithfulness. Reference-fixture tests and fake-prediction contract tests are software
verification, not an evaluation of the deployed model.
