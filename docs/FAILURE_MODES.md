# Failure Modes and Recovery Playbook

Use this playbook during production RAG incidents. Start with the user-visible
symptom, confirm it with logs or metrics, then apply the narrowest recovery
action that restores grounded answers.

## Ingestion Failures

- Symptoms: new documents are missing from search, ingestion exits early, or
  the manifest has stale document versions.
- Likely causes: unreadable source files, PDF parser errors, permission issues,
  or interrupted ingestion jobs.
- Checks: ingestion logs, `data/processed/ingestion_manifest.json`, document
  counts before and after the run, and file permissions.
- Recovery: rerun ingestion for the failed source, quarantine bad files, restore
  the last known-good manifest if needed, then run retrieval smoke tests.
- Escalation: involve the document owner if the source file is corrupt or
  access policy metadata is incomplete.

## Bad Chunking

- Symptoms: answers cite partial sentences, important context is split across
  chunks, or retrieval returns many near-duplicates.
- Likely causes: chunk size too small, overlap too low, malformed source text,
  or tokenizer mismatch.
- Checks: chunk count, average chunk token length, sample chunk previews, and
  `tests/test_chunking.py`.
- Recovery: tune `RAG_CHUNK_SIZE` and `RAG_CHUNK_OVERLAP`, regenerate chunks,
  rebuild the vector store, and rerun golden evals.
- Escalation: add source-specific preprocessing if one document family keeps
  producing poor chunks.

## Missing Citations

- Symptoms: answers are refused despite relevant retrieval, or generated text
  lacks citation ids.
- Likely causes: prompt drift, citation id metadata missing, low retrieval
  quality, or model output not following the citation contract.
- Checks: citation coverage metrics, prompt version, retrieved chunk metadata,
  and `tests/test_citations.py`.
- Recovery: restore the approved citation prompt, reingest documents with
  complete metadata, and rerun generation tests.
- Escalation: switch to extractive answering for affected traffic if citation
  compliance drops below the drift threshold.

## Hallucination

- Symptoms: answers contain claims not present in retrieved chunks or cite
  unrelated chunks.
- Likely causes: weak refusal prompt, irrelevant retrieval results, excessive
  context, or model provider regression.
- Checks: answer/citation pairs, anti-hallucination tests, prompt version,
  `no_answer_rate`, and offline eval failures.
- Recovery: tighten refusal instructions, reduce context to higher-quality
  chunks, enable exact or hybrid mode for lexical queries, and rerun evals.
- Escalation: disable the affected model provider and route to the extractive
  fallback until groundedness recovers.

## Low Recall

- Symptoms: relevant documents exist but are not retrieved in top-k results.
- Likely causes: stale embeddings, poor query wording, missing metadata filters,
  vector DB index drift, or chunking gaps.
- Checks: retrieval traces, top-k candidates, exact search results, sparse/BM25
  lexical results, and golden dataset misses.
- Recovery: rebuild embeddings, increase candidate depth before reranking, use
  hybrid retrieval, and add the failed query to evals.
- Escalation: review vector database choice if recall issues persist at scale
  or require native hybrid scoring.

## Bad Reranking

- Symptoms: relevant candidates are retrieved but reranked below weaker chunks.
- Likely causes: lexical reranker bias, bad score normalization, or insufficient
  candidate depth.
- Checks: pre-rerank candidates, post-rerank order, reranker scores, and
  retrieval tests.
- Recovery: temporarily bypass reranking, increase `candidate_k`, tune lexical
  weights, and add regression tests for the failed query.
- Escalation: evaluate a stronger reranker only after the baseline retrieval
  candidate set is healthy.

## Model Outages

- Symptoms: request timeouts, provider errors, empty model responses, or a spike
  in fallback answers.
- Likely causes: provider downtime, invalid API key, rate limits, or network
  egress failure.
- Checks: API error logs, provider status, latency metrics, configuration
  values, and request volume.
- Recovery: switch provider through runtime settings, reduce traffic, retry
  failed requests with backoff, or use the extractive fallback.
- Escalation: rotate credentials and contact the provider if errors persist
  after configuration and rate-limit checks.

## Vector Database Corruption

- Symptoms: vector store fails to load, query results are empty, or metadata is
  inconsistent with the ingestion manifest.
- Likely causes: interrupted writes, disk issues, incompatible schema change, or
  partial restore.
- Checks: vector DB startup logs, persistence directory integrity, manifest
  versions, and backup timestamps.
- Recovery: stop writes, restore the latest good backup, or rebuild the vector
  store from processed chunks and the manifest.
- Escalation: run a restore drill and consider a managed or transactional store
  if corruption repeats.

## Latency Spikes

- Symptoms: slow API responses, timeouts, queue buildup, or user-visible
  retries.
- Likely causes: high top-k, slow model provider, large context windows,
  reranker cost, vector DB contention, or cold starts.
- Checks: request latency metrics, retrieval timing, generation timing, vector
  DB query logs, and cost metrics.
- Recovery: lower top-k, disable expensive reranking, cache hot queries, scale
  API workers, and route to a faster provider.
- Escalation: split retrieval and generation metrics in dashboards before
  making infrastructure changes.

## Post-Incident Actions

- Add failed production examples to the golden dataset.
- Record the prompt, model, retrieval mode, and document versions involved.
- Update thresholds in drift monitoring if the incident exposed a blind spot.
- Add or update tests before closing the incident.
