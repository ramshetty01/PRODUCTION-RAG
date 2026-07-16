from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Callable
from urllib import error, request


DEFAULT_QUERY = "What evidence is required before vendor onboarding?"


@dataclass(frozen=True)
class LoadSample:
    endpoint: str
    status_code: int
    latency_ms: float
    cached: bool = False
    error: str = ""


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 3)
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile_value
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def summarize(samples: list[LoadSample]) -> dict:
    total = len(samples)
    latencies = [sample.latency_ms for sample in samples]
    status_counts: dict[str, int] = {}
    for sample in samples:
        key = str(sample.status_code)
        status_counts[key] = status_counts.get(key, 0) + 1
    errors = [sample for sample in samples if sample.status_code >= 500 or sample.error]
    rate_limited = [sample for sample in samples if sample.status_code == 429]
    query_samples = [sample for sample in samples if sample.endpoint == "/query"]
    cached = [sample for sample in query_samples if sample.cached]
    return {
        "total_requests": total,
        "status_counts": dict(sorted(status_counts.items())),
        "error_rate": round(len(errors) / total, 4) if total else 0.0,
        "rate_limited_requests": len(rate_limited),
        "latency_ms": {
            "avg": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "cache_hit_rate": round(len(cached) / len(query_samples), 4) if query_samples else 0.0,
        "passed": total > 0 and (len(errors) / total if total else 1.0) <= 0.01 and percentile(latencies, 0.95) <= 3000,
    }


def _request_json(url: str, payload: dict | None = None, headers: dict | None = None) -> LoadSample:
    body = None
    method = "GET"
    request_headers = headers or {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        method = "POST"
        request_headers = {"Content-Type": "application/json", **request_headers}
    started = time.perf_counter()
    endpoint = "/" + url.rstrip("/").split("/", 3)[-1].split("?", 1)[0] if "/" in url.removeprefix("http://").removeprefix("https://") else url
    try:
        req = request.Request(url, data=body, headers=request_headers, method=method)
        with request.urlopen(req, timeout=15) as response:
            raw = response.read().decode("utf-8")
            latency_ms = (time.perf_counter() - started) * 1000
            parsed = json.loads(raw) if raw and raw.strip().startswith("{") else {}
            return LoadSample(endpoint=endpoint, status_code=response.status, latency_ms=latency_ms, cached=bool(parsed.get("cached")))
    except error.HTTPError as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return LoadSample(endpoint=endpoint, status_code=exc.code, latency_ms=latency_ms, error=str(exc))
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return LoadSample(endpoint=endpoint, status_code=0, latency_ms=latency_ms, error=type(exc).__name__)


def run_load_test(
    base_url: str,
    requests_per_endpoint: int = 10,
    concurrency: int = 4,
    query: str = DEFAULT_QUERY,
    api_key: str = "",
    request_func: Callable[[str, dict | None, dict | None], LoadSample] = _request_json,
) -> dict:
    base_url = base_url.rstrip("/")
    headers = {"X-API-Key": api_key} if api_key else {}
    work: list[tuple[str, dict | None, dict | None]] = []
    for _ in range(requests_per_endpoint):
        work.append((f"{base_url}/health", None, None))
        work.append((f"{base_url}/metrics", None, None))
        work.append(
            (
                f"{base_url}/query",
                {"query": query, "retrieval_mode": "hybrid", "top_k": 4},
                headers,
            )
        )

    samples: list[LoadSample] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(request_func, url, payload, request_headers) for url, payload, request_headers in work]
        for future in as_completed(futures):
            samples.append(future.result())
    return {"summary": summarize(samples), "samples": [asdict(sample) for sample in samples]}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a lightweight load test against the Production RAG API.")
    parser.add_argument("base_url", help="Base URL, for example http://localhost:8000")
    parser.add_argument("--requests-per-endpoint", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_load_test(
        args.base_url,
        requests_per_endpoint=args.requests_per_endpoint,
        concurrency=args.concurrency,
        query=args.query,
        api_key=args.api_key,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
