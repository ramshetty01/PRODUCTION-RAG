# Portfolio Guide

## Five-Minute Reviewer Path

1. Open `/demo` and ask "What evidence is required before vendor onboarding?"
2. Inspect the answer, citations, retrieval trace, auth badge, and quality panel.
3. Ask "What is the vendor bank account number?" and confirm the system refuses unsupported sensitive requests.
4. Read [Architecture Diagram](ARCHITECTURE_DIAGRAM.md).
5. Scan [Testing Strategy](TESTING.md), [Deployment](DEPLOYMENT.md), and [Golden Evaluation Dataset](../evals/README.md).

## What This Proves

- Production RAG fundamentals: ingestion, chunking, embeddings, vector search, hybrid retrieval, reranking, citations, and refusal behavior.
- AI engineering maturity: versioned prompts, golden evals, CI gates, online feedback, drift signals, load testing, and deployment docs.
- Backend production readiness: FastAPI, auth, rate limiting, cache isolation, managed vector backend option, OpenTelemetry spans, Docker, Render, and Kubernetes artifacts.
- Portfolio clarity: the demo exposes answer quality, evidence, auth subject, roles, request trace, and evaluation metrics in one screen.

## How I Built It

The project started as a local RAG pipeline and was hardened issue by issue. The core path ingests enterprise documents, chunks them with stable metadata, stores vectors in Chroma or Qdrant, retrieves authorized chunks, optionally combines lexical and vector search, reranks candidates, and generates only answers supported by retrieved evidence.

The production layer adds API contracts, prompt versioning, citation enforcement, auth-aware cache keys, feedback monitoring, evaluation dashboards, reliability tests, deployment manifests, and observability hooks.

## Key Tradeoffs

- Chroma remains the local default because it keeps tests and demos reproducible. Qdrant is available when a remote managed index is needed.
- The deterministic eval runner is lightweight enough for CI. Ragas mode is still supported for deeper model-backed evaluation when credentials and dependencies are available.
- The demo includes API-key/JWT-style credential entry, but real deployments should rely on upstream identity providers and rotate secrets outside Git.
- OpenTelemetry is optional so local development does not require a collector. Production can enable OTLP export with environment variables.
- The frontend is a demo console, not a full SaaS app. It is designed to reveal production AI behaviors quickly: evidence, roles, quality, latency, and request state.

## Resume Bullets

- Built a production-style RAG platform with FastAPI, Chroma/Qdrant vector backends, hybrid retrieval, reranking, citation enforcement, and refusal behavior for unsupported claims.
- Implemented enterprise AI quality gates with a verified golden dataset, deterministic/Ragas-compatible evaluation, quality dashboard, and CI-ready faithfulness thresholds.
- Added production controls including API key/JWT auth, role-aware retrieval, cache isolation, rate limiting, OpenTelemetry tracing, Prometheus metrics, and load-test reporting.
- Packaged the system for deployment with Docker, Render, Kubernetes manifests, environment-based configuration, managed vector store setup, and reviewer-focused demo documentation.

## Demo Assets

- [Demo Video Script](DEMO_VIDEO_SCRIPT.md)
- [Architecture Diagram](ARCHITECTURE_DIAGRAM.md)
- [Demo Frontend Screenshot Asset](assets/demo-frontend.svg)
