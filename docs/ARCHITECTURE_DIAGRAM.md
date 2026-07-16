# Architecture Diagram

```mermaid
flowchart LR
    A[Enterprise corpus\nPDF, Markdown, text] --> B[Ingestion manifest\nhash and version]
    B --> C[Token chunking\nmetadata and roles]
    C --> D[Embedding provider]
    D --> E{Vector backend}
    E -->|local default| F[ChromaDB]
    E -->|managed option| G[Qdrant]
    F --> H[Retriever]
    G --> H
    H --> I[Hybrid search\nBM25 plus vector]
    I --> J[Reranker]
    J --> K[Prompt contract\nretrieved evidence only]
    K --> L[LLM or extractive generator]
    L --> M[Citation enforcement\nrefuse unsupported claims]
    M --> N[FastAPI response]
    N --> O[Demo UI\nanswer, evidence, auth, quality]
    N --> P[Metrics, logs, OTEL spans]
    Q[Golden eval dataset] --> R[Evaluation dashboard]
    R --> O
```

## Reviewer Reading Order

1. Start with the browser demo at `/demo`.
2. Read [Portfolio Guide](PORTFOLIO.md) for the story, tradeoffs, and resume bullets.
3. Read [Architecture](ARCHITECTURE.md) for module ownership and runtime state.
4. Read [Testing Strategy](TESTING.md) and [Golden Evaluation Dataset](../evals/README.md) for quality gates.
5. Read [Deployment](DEPLOYMENT.md) for Docker, Render, Redis, Qdrant, and OpenTelemetry setup.
