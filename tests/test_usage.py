from src.rag.usage import UsageEvent, append_usage, load_usage, usage_summary


def test_usage_events_round_trip_and_aggregate_by_subject_org_workspace(tmp_path):
    path = tmp_path / "usage.jsonl"
    append_usage(UsageEvent("req-1", "user-a", "org-a", "workspace-a", 10, 5, 0.01), path)
    append_usage(UsageEvent("req-2", "user-a", "org-a", "workspace-b", 20, 10, 0.02), path)

    events = load_usage(path)
    summary = usage_summary(events)
    scoped = usage_summary(events, workspace_id="workspace-a")

    assert len(events) == 2
    assert summary["total_requests"] == 2
    assert summary["total_tokens"] == 45
    assert summary["estimated_cost"] == 0.03
    assert summary["by_user"]["user-a"]["tokens"] == 45
    assert summary["by_org"]["org-a"]["requests"] == 2
    assert scoped["total_requests"] == 1
    assert scoped["by_workspace"]["workspace-a"]["estimated_cost"] == 0.01
