from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_USAGE_LOG = Path("logs/usage.jsonl")


@dataclass
class UsageEvent:
    request_id: str
    subject: str
    org_id: str
    workspace_id: str
    prompt_tokens: int
    answer_tokens: int
    estimated_cost: float
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()


def append_usage(event: UsageEvent, path: str | Path = DEFAULT_USAGE_LOG) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")


def load_usage(path: str | Path = DEFAULT_USAGE_LOG) -> list[UsageEvent]:
    path = Path(path)
    if not path.exists():
        return []
    return [UsageEvent(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def usage_summary(events: list[UsageEvent], workspace_id: str | None = None) -> dict:
    scoped = [event for event in events if not workspace_id or event.workspace_id == workspace_id]
    by_subject: dict[str, dict] = {}
    by_org: dict[str, dict] = {}
    by_workspace: dict[str, dict] = {}
    for event in scoped:
        for bucket, key in [(by_subject, event.subject), (by_org, event.org_id), (by_workspace, event.workspace_id)]:
            row = bucket.setdefault(key, {"requests": 0, "tokens": 0, "estimated_cost": 0.0})
            row["requests"] += 1
            row["tokens"] += event.prompt_tokens + event.answer_tokens
            row["estimated_cost"] = round(row["estimated_cost"] + event.estimated_cost, 6)
    return {
        "total_requests": len(scoped),
        "total_tokens": sum(event.prompt_tokens + event.answer_tokens for event in scoped),
        "estimated_cost": round(sum(event.estimated_cost for event in scoped), 6),
        "by_user": by_subject,
        "by_org": by_org,
        "by_workspace": by_workspace,
    }
