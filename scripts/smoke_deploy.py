from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _request(url: str, *, method: str = "GET", payload: dict | None = None, timeout: int = 30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, response.headers.get("Content-Type", ""), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with {exc.code}: {body}") from exc


def smoke(base_url: str) -> list[str]:
    base_url = base_url.rstrip("/")
    checks = []

    status, _, body = _request(f"{base_url}/health")
    if status != 200 or '"ok"' not in body:
        raise RuntimeError("/health did not return ok")
    checks.append("/health")

    status, content_type, body = _request(f"{base_url}/demo")
    if status != 200 or "text/html" not in content_type or "Production RAG Demo Console" not in body:
        raise RuntimeError("/demo did not return the demo frontend")
    checks.append("/demo")

    status, content_type, body = _request(f"{base_url}/metrics")
    if status != 200 or "text/plain" not in content_type or "rag_api_requests_total" not in body:
        raise RuntimeError("/metrics did not return Prometheus text")
    checks.append("/metrics")

    status, _, body = _request(
        f"{base_url}/query",
        method="POST",
        payload={
            "query": "What evidence is required before vendor onboarding?",
            "retrieval_mode": "hybrid",
            "top_k": 4,
        },
        timeout=90,
    )
    payload = json.loads(body)
    if status != 200 or "answer" not in payload or "request_id" not in payload:
        raise RuntimeError("/query did not return a valid RAG response")
    checks.append("/query")

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a deployed Production RAG service.")
    parser.add_argument("base_url", help="Base URL, for example https://production-rag-demo.onrender.com")
    args = parser.parse_args()

    checks = smoke(args.base_url)
    print("Deployment smoke test passed:")
    for check in checks:
        print(f"- {check}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
