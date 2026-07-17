# Reranking

Reranking improves precision by rescoring retrieved chunks after the first
vector, keyword, or hybrid candidate pass. `RAG_RETRIEVAL_MODE=reranked` is the
default production retrieval mode, using hybrid retrieval for candidates before
reranking the final Top K. The default test-safe reranker is lexical and
deterministic. Production runs can opt into a real cross-encoder reranker from
Sentence Transformers.

## Providers

- `lexical`: local fallback based on query-term overlap and exact phrase bonus.
- `cross_encoder`: loads a Sentence Transformers `CrossEncoder` model and scores
  each query/chunk pair together.

## Configuration

Set these environment variables in `.env`:

```bash
RAG_RETRIEVAL_MODE=reranked
RAG_RERANKER_PROVIDER=cross_encoder
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG_RERANKER_ALLOW_FALLBACK=true
```

Keep fallback enabled for demos and CI-like environments where model downloads
may be unavailable. Disable fallback in production if startup should fail loudly
when the configured reranker cannot load.

## Tradeoffs

Cross-encoders usually improve top-k precision because they evaluate the query
and candidate chunk as a pair. The cost is higher latency and model loading
overhead. Use the lexical reranker for unit tests, offline fallback, and very
small local demos; use the cross-encoder path when retrieval quality matters
more than reranking latency.
