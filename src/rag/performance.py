from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass
class QueryCache:
    values: dict[str, dict] = field(default_factory=dict)

    def key(self, query: str, top_k: int, filters: dict | None = None) -> str:
        raw = f"{query}|{top_k}|{filters or {}}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, query: str, top_k: int, filters: dict | None = None) -> dict | None:
        return self.values.get(self.key(query, top_k, filters))

    def set(self, query: str, top_k: int, value: dict, filters: dict | None = None) -> None:
        self.values[self.key(query, top_k, filters)] = value


DEFAULT_LATENCY_BUDGETS_MS = {
    "ingestion": 30_000,
    "retrieval": 2_000,
    "reranking": 2_000,
    "generation": 10_000,
}


def check_latency_budget(stage: str, latency_ms: float, budgets: dict | None = None) -> bool:
    budgets = budgets or DEFAULT_LATENCY_BUDGETS_MS
    return latency_ms <= budgets[stage]


def estimate_llm_cost(token_usage: dict, cost_per_1k_tokens: float = 0.0) -> float:
    total_tokens = sum(int(value) for value in token_usage.values())
    return round((total_tokens / 1000) * cost_per_1k_tokens, 6)


def call_with_retries(
    func: Callable[[], T],
    retries: int = 2,
    timeout_seconds: float | None = None,
    sleep_seconds: float = 0.0,
) -> T:
    start = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        if timeout_seconds is not None and time.perf_counter() - start > timeout_seconds:
            raise TimeoutError("external call timed out") from last_error
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                raise
            if sleep_seconds:
                time.sleep(sleep_seconds)
    raise RuntimeError("unreachable retry state")
