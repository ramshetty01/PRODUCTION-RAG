from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from src.rag.security import redact_pii


DEFAULT_AUDIT_LOG = Path("logs/audit.jsonl")


def append_audit_event(event: dict, path: str | Path = DEFAULT_AUDIT_LOG) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        **event,
        "timestamp": event.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "query": redact_pii(str(event.get("query", ""))),
        "answer": redact_pii(str(event.get("answer", ""))),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def load_audit_events(path: str | Path = DEFAULT_AUDIT_LOG, limit: int = 100) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines[-limit:] if line.strip()]
    return list(reversed(events))


def audit_events_csv(events: list[dict]) -> str:
    output = io.StringIO()
    fields = ["timestamp", "user", "query", "retrieval_ids", "answer", "citations", "model", "latency_ms"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for event in events:
        row = {field: event.get(field, "") for field in fields}
        row["retrieval_ids"] = "|".join(event.get("retrieval_ids", []))
        row["citations"] = "|".join(event.get("citations", []))
        writer.writerow(row)
    return output.getvalue()
