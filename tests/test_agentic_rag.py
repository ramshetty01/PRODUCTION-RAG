from src.rag.advanced.agentic_rag import (
    decompose_query,
    plan_agentic_retrieval,
    route_subquery,
    should_use_agentic_rag,
)


def test_decompose_query_splits_multi_step_question():
    assert decompose_query("Find ZX-144 and compare it with the deployment guide") == [
        "Find ZX-144",
        "compare it with the deployment guide",
    ]


def test_route_subquery_prefers_exact_for_identifiers():
    assert route_subquery("Find ZX-144") == "exact"
    assert route_subquery("How does deployment work?") == "hybrid"
    assert route_subquery("What is a runner") == "semantic"


def test_agentic_plan_is_inspectable_and_guarded():
    trace = plan_agentic_retrieval("Find ZX-144 and compare it with deployment")

    assert trace.routes == ["exact", "hybrid"]
    assert [step.name for step in trace.steps] == ["decompose", "route", "guardrail"]
    assert "cited context" in trace.steps[-1].output


def test_agentic_rag_gate_keeps_simple_questions_on_core_path():
    assert should_use_agentic_rag("What is a runner?") is False
    assert should_use_agentic_rag("Compare runners and workflows") is True
