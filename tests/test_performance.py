import pytest

from src.rag.performance import QueryCache, call_with_retries, check_latency_budget, estimate_llm_cost


def test_query_cache_uses_query_top_k_and_filters():
    cache = QueryCache()
    cache.set("what is a runner", 2, {"answer": "cached"}, {"document_id": "docs"})

    assert cache.get("what is a runner", 2, {"document_id": "docs"}) == {"answer": "cached"}
    assert cache.get("what is a runner", 1, {"document_id": "docs"}) is None


def test_latency_budget_and_cost_estimate():
    assert check_latency_budget("retrieval", 100)
    assert not check_latency_budget("retrieval", 10_000)
    assert estimate_llm_cost({"prompt_tokens": 750, "answer_tokens": 250}, cost_per_1k_tokens=0.01) == 0.01


def test_call_with_retries_retries_then_succeeds():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("temporary")
        return "ok"

    assert call_with_retries(flaky, retries=2) == "ok"
    assert calls["count"] == 2


def test_call_with_retries_raises_after_retries():
    with pytest.raises(RuntimeError):
        call_with_retries(lambda: (_ for _ in ()).throw(RuntimeError("fail")), retries=1)
