from scripts.load_test import LoadSample, UploadPayload, run_load_test, summarize


def test_summarize_reports_latency_error_cache_and_rate_limit_metrics():
    samples = [
        LoadSample("/health", 200, 10),
        LoadSample("/metrics", 200, 20),
        LoadSample("/query", 200, 30, cached=False),
        LoadSample("/query", 200, 40, cached=True),
        LoadSample("/query", 429, 50),
        LoadSample("/query", 500, 60, error="server"),
    ]

    summary = summarize(samples)

    assert summary["total_requests"] == 6
    assert summary["status_counts"] == {"200": 4, "429": 1, "500": 1}
    assert summary["error_rate"] == 0.1667
    assert summary["rate_limited_requests"] == 1
    assert summary["latency_ms"]["p50"] == 35.0
    assert summary["latency_ms"]["p95"] == 57.5
    assert summary["cache_hit_rate"] == 0.25
    assert summary["passed"] is False


def test_run_load_test_exercises_health_metrics_and_query():
    calls = []

    def fake_request(url, payload=None, headers=None):
        calls.append((url, payload, headers))
        if url.endswith("/query"):
            return LoadSample("/query", 200, 25, cached=len([call for call in calls if call[0].endswith("/query")]) > 1)
        if url.endswith("/metrics"):
            return LoadSample("/metrics", 200, 10)
        return LoadSample("/health", 200, 5)

    report = run_load_test(
        "http://localhost:8000",
        requests_per_endpoint=2,
        concurrency=2,
        api_key="public-key",
        request_func=fake_request,
    )

    assert report["profile"] == "standard"
    assert report["summary"]["total_requests"] == 12
    assert report["summary"]["status_counts"] == {"200": 12}
    assert report["summary"]["cache_hit_rate"] == 0.5
    assert report["summary"]["passed"] is True
    assert any(call[0].endswith("/health") for call in calls)
    assert any(call[0].endswith("/metrics") for call in calls)
    assert any(call[0].endswith("/upload") and isinstance(call[1], UploadPayload) for call in calls)
    assert any("/index-status?workspace_id=load-test" in call[0] for call in calls)
    assert any(call[0].endswith("/query/stream") for call in calls)
    query_calls = [call for call in calls if call[0].endswith("/query")]
    assert query_calls[0][1]["retrieval_mode"] == "hybrid"
    assert query_calls[0][1]["workspace_id"] == "load-test"
    assert query_calls[0][2] == {"X-API-Key": "public-key"}


def test_smoke_profile_is_ci_safe_and_covers_core_paths():
    calls = []

    def fake_request(url, payload=None, headers=None):
        calls.append((url, payload, headers))
        return LoadSample("/ok", 200, 5)

    report = run_load_test(
        "http://localhost:8000",
        requests_per_endpoint=25,
        concurrency=8,
        profile="smoke",
        request_func=fake_request,
    )

    assert report["profile"] == "smoke"
    assert report["requests_per_endpoint"] == 1
    assert report["concurrency"] == 2
    assert len(calls) == 6
    assert {call[0].rsplit("/", 1)[-1].split("?", 1)[0] for call in calls} >= {"health", "metrics", "upload", "index-status", "query", "stream"}


def test_abuse_profile_adds_oversized_upload_and_repeated_query_pressure():
    calls = []

    def fake_request(url, payload=None, headers=None):
        calls.append((url, payload, headers))
        return LoadSample("/ok", 200, 5)

    run_load_test(
        "http://localhost:8000",
        requests_per_endpoint=1,
        profile="abuse",
        large_upload_bytes=123,
        request_func=fake_request,
    )

    uploads = [call[1] for call in calls if call[0].endswith("/upload")]
    query_calls = [call for call in calls if call[0].endswith("/query")]
    assert len(uploads) == 2
    assert len(uploads[0].content) < len(uploads[1].content)
    assert len(uploads[1].content) == 123
    assert len(query_calls) == 2
