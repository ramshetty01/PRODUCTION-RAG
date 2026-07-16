# Demo Video Script

Target length: 3 to 4 minutes.

## 0:00 - 0:20 Opening

Show `/demo`.

Say: "This is a production-style RAG system over an enterprise security and vendor-risk corpus. I built it to show the full lifecycle: ingestion, retrieval quality, citations, access control, evaluation, deployment, and observability."

## 0:20 - 0:55 Grounded Answer

Use the default question: "What evidence is required before vendor onboarding?"

Show the cited answer, citations panel, retrieval mode, returned chunks, request ID, and quality dashboard.

Say: "The important part is not just the answer. The UI shows the retrieved evidence, the chunk IDs used as citations, and the evaluation gate that protects quality."

## 0:55 - 1:25 Refusal Behavior

Ask: "What is the vendor bank account number?"

Show the refusal response.

Say: "Unsupported sensitive requests are refused when the retrieved context does not support the answer. This is citation enforcement, not a chatbot demo."

## 1:25 - 2:00 Auth And Protected Retrieval

Set `RAG_API_KEYS=public-key:public,admin-key:public|admin` before recording.

In `/demo`, choose Public and ask: "Can I retrieve protected payroll data?"

Then choose Admin and ask the same question.

Say: "The same query changes behavior based on authenticated roles. Public users cannot retrieve admin-only chunks. Admin users can retrieve protected chunks when the backend authorizes them."

## 2:00 - 2:35 Evaluation And Reliability

Show `/evaluation` JSON or the quality panel.

Run:

```bash
python evals/run_ragas.py --config configs/settings.toml
python scripts/load_test.py http://localhost:8000 --requests-per-endpoint 25 --concurrency 8
```

Say: "Every quality claim is backed by tests or a report: faithfulness, citation coverage, refusal accuracy, latency, cache behavior, and rate-limit signals."

## 2:35 - 3:20 Production Architecture

Open [Architecture Diagram](ARCHITECTURE_DIAGRAM.md).

Say: "The local default uses ChromaDB, but the vector backend can switch to Qdrant. The API supports Redis cache/rate limiting, API key or JWT auth, OpenRouter-compatible LLMs, OpenTelemetry spans, Docker, Render, and Kubernetes deployment assets."

## 3:20 - 3:45 Close

Open [Portfolio Guide](PORTFOLIO.md).

Say: "This project is designed to be reviewed quickly by HR and deeply by engineering. The README gives the quick path; the docs show architecture, tradeoffs, tests, deployment, and resume-ready impact."
