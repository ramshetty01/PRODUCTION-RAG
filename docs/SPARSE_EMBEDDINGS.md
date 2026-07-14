# Sparse Embeddings

Sparse embeddings represent text as weighted lexical dimensions instead of dense
semantic vectors. They are useful when retrieval must preserve exact terms such
as error codes, product names, identifiers, configuration keys, and uncommon
domain vocabulary.

## Options Considered

- BM25: strong lexical baseline, already local and deterministic in this repo.
- TF-IDF sparse vectors: simple weighted sparse representation that can be
  inspected and tested without an external service.
- SPLADE-style learned sparse embeddings: stronger semantic expansion, but adds
  model hosting, latency, and dependency cost.
- Hosted hybrid search engines: useful later if the vector database provides
  sparse-vector fields and hybrid scoring natively.

## Current Choice

The project keeps BM25 as the default lexical retriever and adds an optional
local TF-IDF sparse path in `src/rag/advanced/sparse_embeddings.py`. This is a
minimal implementation intended for exact and lexical queries where dense
vectors may under-rank rare terms. It is deterministic, requires no new runtime
service, and can be enabled through the `sparse` retrieval mode.

BM25 remains enough for most lexical matching today. Sparse TF-IDF is useful
when the operator wants explicit sparse-vector scoring, score inspection, or a
future migration path toward learned sparse vectors.

## Configuration

Sparse retrieval is optional. Call `retrieve_by_mode(..., mode="sparse")` or
`retrieve_sparse_chunks(...)` with the candidate documents. Existing semantic,
hybrid, and exact modes are unchanged.

## Demonstrated Lexical Query

The test suite includes a lexical identifier query for `ZX-144`. The sparse
retriever ranks the chunk containing that identifier first without requiring a
vector database. This demonstrates the benefit for rare identifiers while
confirming BM25 remains a suitable default lexical baseline.
