# Production Alerting

Scrape `/metrics` from Prometheus, Grafana Cloud, Datadog, or any compatible
collector. Route these alerts to the on-call channel for the deployed service.

```yaml
groups:
  - name: production-rag
    rules:
      - alert: RAGIngestionFailures
        expr: rag_ingestion_failures_total > 0
        for: 10m
        labels:
          severity: page
        annotations:
          summary: "RAG ingestion is failing"
          description: "Failed jobs or failed document records require reindexing."

      - alert: RAGLLMProviderErrors
        expr: increase(rag_llm_provider_failures_total[10m]) > 2
        for: 5m
        labels:
          severity: page
        annotations:
          summary: "LLM provider failures are elevated"
          description: "Check provider health, API key validity, fallback providers, and model timeout settings."

      - alert: RAGHighLatency
        expr: rag_api_request_latency_ms_avg > 3000
        for: 10m
        labels:
          severity: warn
        annotations:
          summary: "RAG API latency is high"
          description: "Inspect retrieval, reranker, and LLM latency before scaling."

      - alert: RAGCostSpike
        expr: increase(rag_llm_estimated_cost_total[1h]) > 10
        for: 5m
        labels:
          severity: warn
        annotations:
          summary: "RAG LLM cost is spiking"
          description: "Review traffic, prompt size, top_k, reranking, and model selection."

      - alert: RAGStorageUsageHigh
        expr: rag_storage_usage_bytes > 5000000000
        for: 30m
        labels:
          severity: warn
        annotations:
          summary: "RAG local storage usage is high"
          description: "Run retention, move uploads to object storage, or compact the vector index."

      - alert: RAGAuthFailures
        expr: increase(rag_auth_failures_total[10m]) > 20
        for: 5m
        labels:
          severity: warn
        annotations:
          summary: "Authentication failures are elevated"
          description: "Check API key/JWT configuration and possible credential misuse."
```

Tune thresholds per environment. The local defaults are intentionally low enough
to validate alert wiring before production traffic.
