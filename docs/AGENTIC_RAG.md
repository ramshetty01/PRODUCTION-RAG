# Agentic RAG Exploration

Agentic RAG can help when a user question requires decomposition, route
selection, or tool-using retrieval. It should not replace the core RAG path for
simple factual questions because every extra planning step adds latency,
complexity, and audit surface.

## Use Cases

- Multi-step questions that ask for two or more facts before synthesis.
- Comparison questions across documents, versions, or sections.
- Queries that mix exact identifiers with broad semantic context.
- Operational investigations where the system should first find a relevant
  document family and then retrieve supporting chunks.

## Simple RAG vs Agentic RAG

Simple RAG is preferred for single-hop questions. It is easier to test, cheaper
to run, and has fewer opportunities to retrieve irrelevant context.

Agentic RAG is justified only when query decomposition or route selection
improves evidence quality. The current prototype plans retrieval steps but keeps
generation under the existing citation and refusal rules.

## Prototype

`src/rag/advanced/agentic_rag.py` provides a deterministic planning prototype:

- `decompose_query` splits multi-part questions.
- `route_subquery` chooses `exact`, `hybrid`, or `semantic` retrieval.
- `plan_agentic_retrieval` returns an inspectable trace with every step.
- `should_use_agentic_rag` gates the workflow to multi-step or comparison
  questions.

This is LangGraph-style in shape because it records explicit nodes and state,
but it avoids adding a graph runtime until the project has production evidence
that the extra dependency is worth it.

## Guardrails

- The planner is read-only and cannot mutate documents or configuration.
- Every step is captured in `AgenticRAGTrace`.
- Final answers still require retrieved citations.
- Refusal behavior remains unchanged when evidence is missing.
- Agentic mode should be disabled by default until online and offline evals show
  better quality than simple or hybrid retrieval.

## Decision

Keep agentic RAG as an optional exploration path for now. The core reliability
priority remains grounded retrieval, citation enforcement, and measurable eval
quality. Promote the prototype only after it improves multi-step benchmark cases
without hurting latency or citation quality.
