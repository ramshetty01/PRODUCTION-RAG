# Vector Database Selection

The project uses ChromaDB for the current production-RAG baseline. Chroma keeps
local development simple, persists vectors on disk, and works well for the
current single-service deployment model. This document defines when that choice
should change.

## Selection Criteria

- Local development: easy setup, deterministic tests, no mandatory cloud
  account.
- Persistence: durable indexes, backup-friendly data layout, and restore
  support.
- Metadata filtering: document, version, tenant, and access-role filters.
- Scaling: index size, write throughput, read latency, and horizontal growth.
- Cost: local runtime cost, hosted service cost, and operational staffing.
- Deployment: container support, managed options, network boundaries, and CI
  compatibility.
- Backups: point-in-time backup, restore drills, and corruption recovery.
- Access control: tenant isolation, auth integration, and least-privilege
  service credentials.

## Options

| Option | Strengths | Tradeoffs | Best Fit |
| --- | --- | --- | --- |
| ChromaDB | Fast local setup, simple persistence, Python-native workflow | Limited managed production controls compared with dedicated services | Current local and small-team deployments |
| Qdrant | Strong filtering, payload indexes, hybrid search support, self-hosted or managed | Adds a separate service and backup lifecycle | Medium-scale production with strict filters |
| Weaviate | Rich schema, hybrid search, modules, managed cloud option | More operational complexity and schema planning | Teams needing semantic plus structured search at scale |
| Pinecone | Managed scaling, operationally simple, production SLAs | External service cost and vendor dependency | High-scale managed vector search with minimal ops |
| pgvector | Keeps vectors beside relational data, transactional backups, SQL access control | Requires Postgres tuning and may lag specialist engines at large scale | Apps already standardized on Postgres |

## Current Decision

Use ChromaDB by default. It is the best fit while the project emphasizes local
development, reproducible ingestion, simple backups, and low operational cost.
The runtime vector backend is pluggable, and Qdrant is the supported managed
or remote-production option when deployment needs a service outside the API
process.

## Runtime Configuration

Local default:

```bash
RAG_VECTOR_BACKEND=chroma
RAG_VECTOR_DB_PATH=./chroma_db
RAG_VECTOR_COLLECTION=rag_chunks
```

Managed Qdrant:

```bash
RAG_VECTOR_BACKEND=qdrant
RAG_VECTOR_COLLECTION=rag_chunks
RAG_QDRANT_URL=https://example.qdrant.tech
RAG_QDRANT_API_KEY=replace-me
```

The Qdrant backend requires the optional `langchain-qdrant` package. Keep
Chroma for local CI and demos unless you are validating a real remote index.
Uploaded chunks preserve `workspace_id`, `document_id`, `document_version`, and
`access_roles` metadata. Retrieval filters by workspace and role in the app
boundary, and deletion uses the same metadata filter against Chroma or Qdrant.

## Migration Triggers

Move away from ChromaDB when one or more of these conditions is true:

- Index size or query traffic causes sustained latency beyond service targets.
- Multiple tenants require stronger isolation than application-side filtering.
- Operators need managed backups, point-in-time recovery, or formal SLAs.
- Hybrid sparse/vector retrieval needs native database-level scoring.
- Deployment requires a remote vector service shared by multiple API workers.
- Access control must be enforced by the storage layer rather than by the app.

## Migration Risks

- Embedding dimensions and distance metrics may not map one-to-one between
  stores.
- Metadata filter syntax and type handling differ across engines.
- Reindexing can temporarily reduce recall if chunk ids or document versions
  are not preserved.
- Backup and restore procedures must be rewritten and tested before cutover.
- Ranking changes require offline and online evaluation before production use.

## Recommended Path

Keep ChromaDB as the default. If migration triggers appear, run a proof of
concept against Qdrant and pgvector first because they preserve self-hosting and
backup control. Consider Pinecone or Weaviate when managed scaling and service
operations are more important than local simplicity.
