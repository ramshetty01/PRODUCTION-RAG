from __future__ import annotations

import re
from dataclasses import dataclass, field


LEXICAL_PATTERN = re.compile(r"\b[A-Z]{2,}[-_]\d+\b|\b\w+\.\w+\b")


@dataclass(frozen=True)
class AgentStep:
    name: str
    input: str
    output: str


@dataclass
class AgenticRAGTrace:
    question: str
    subqueries: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)

    def add_step(self, name: str, input_text: str, output: str) -> None:
        self.steps.append(AgentStep(name=name, input=input_text, output=output))


def decompose_query(question: str) -> list[str]:
    parts = [
        part.strip(" ?.")
        for part in re.split(r"\b(?:and|then|also)\b|[;?]", question, flags=re.IGNORECASE)
        if part.strip(" ?.")
    ]
    return parts or [question.strip()]


def route_subquery(subquery: str) -> str:
    if LEXICAL_PATTERN.search(subquery):
        return "exact"
    lowered = subquery.lower()
    if any(term in lowered for term in ("compare", "difference", "why", "how")):
        return "hybrid"
    return "semantic"


def plan_agentic_retrieval(question: str) -> AgenticRAGTrace:
    trace = AgenticRAGTrace(question=question)
    trace.subqueries = decompose_query(question)
    trace.add_step("decompose", question, " | ".join(trace.subqueries))
    trace.routes = [route_subquery(subquery) for subquery in trace.subqueries]
    trace.add_step("route", " | ".join(trace.subqueries), " | ".join(trace.routes))
    trace.add_step("guardrail", question, "read-only retrieval plan; generation still requires cited context")
    return trace


def should_use_agentic_rag(question: str) -> bool:
    subqueries = decompose_query(question)
    if len(subqueries) > 1:
        return True
    lowered = question.lower()
    return any(term in lowered for term in ("compare", "step by step", "across documents", "first find"))
