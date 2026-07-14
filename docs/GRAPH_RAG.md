# Graph RAG Exploration

Graph RAG represents relationships between chunks, documents, sections, and
entities so retrieval can move from an initial evidence chunk to related
context. It can help when the answer depends on connected facts rather than a
single matching chunk.

## Relationships Worth Modeling

- Document lineage: chunks from the same `document_id` or source file.
- Section adjacency: chunks from the same section or neighboring process area.
- Shared operational entities: error codes, workflow concepts, service names,
  configuration keys, and incident terms.
- Version relationships: old and new versions of the same document.
- Access-control boundaries: relationships should not bypass metadata and role
  filtering.

## When Graph RAG Helps

Graph retrieval is useful when users ask cross-document or multi-hop questions,
such as comparing a policy with an incident playbook, following a workflow from
trigger to job to runner, or expanding an exact error-code match to related
recovery guidance. It can improve recall when the first retrieved chunk is
correct but incomplete.

## When It Is Unnecessary

Graph RAG is unnecessary for simple single-hop questions, exact definitions, or
small document sets where hybrid retrieval already returns complete context. It
also adds graph-construction cost, relationship-quality risk, and another layer
of ranking behavior to evaluate.

## Prototype

`src/rag/advanced/graph_rag.py` implements an exploratory in-memory graph:

- metadata edges connect chunks with the same document, source, or section.
- shared-term edges connect chunks with enough lexical overlap.
- graph expansion starts from seed retrieval results and adds related chunks.
- every expanded result records the relation in the `source` field, such as
  `graph:document_id` or `graph:shared_terms`.

The prototype intentionally avoids a graph database. It is small enough to test
locally and helps decide whether richer relationship extraction is justified.

## Comparison With Existing Retrieval

- Vector retrieval finds semantically similar chunks.
- Hybrid retrieval combines semantic and lexical evidence.
- Graph retrieval adds neighboring context after a seed match is already found.

Graph RAG should be evaluated as a supplement to hybrid retrieval, not a
replacement. A quality win would look like better answers for multi-hop
questions without lower citation quality or unacceptable latency.

## Decision

Keep Graph RAG exploratory for now. The project has useful metadata
relationships, but production promotion should wait until offline and online
evals show that graph expansion improves multi-step answers. The current
prototype is sufficient for targeted experiments tied to project use cases, not
hype-driven architecture expansion.
