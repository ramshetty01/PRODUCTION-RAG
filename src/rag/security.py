from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from pathlib import Path


PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|system|developer) instructions", re.I),
    re.compile(r"reveal (the )?(system prompt|developer message|secrets?)", re.I),
    re.compile(r"you are now", re.I),
]
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d -]{7,}\d)\b")


def contains_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in PROMPT_INJECTION_PATTERNS)


def redact_pii(text: str) -> str:
    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    return PHONE_PATTERN.sub("[REDACTED_PHONE]", text)


def validate_query(query: str, max_length: int = 2000) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")
    if len(query) > max_length:
        raise ValueError("query is too long")
    if contains_prompt_injection(query):
        raise ValueError("query contains unsafe instructions")
    return redact_pii(query)


def validate_path(path: str | Path, allowed_root: str | Path = ".") -> Path:
    allowed_root = Path(allowed_root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = allowed_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"path is outside allowed root: {path}") from exc
    return resolved


class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        window = self.requests[key]
        while window and now - window[0] > self.window_seconds:
            window.popleft()
        if len(window) >= self.max_requests:
            return False
        window.append(now)
        return True


class RedisRateLimiter:
    def __init__(self, redis_url: str, max_requests: int = 60, window_seconds: int = 60, prefix: str = "rag:rate"):
        from redis import Redis

        self.client = Redis.from_url(redis_url, decode_responses=True)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.prefix = prefix

    def allow(self, key: str) -> bool:
        now = int(time.time())
        bucket = now // self.window_seconds
        redis_key = f"{self.prefix}:{key}:{bucket}"
        count = self.client.incr(redis_key)
        if count == 1:
            self.client.expire(redis_key, self.window_seconds)
        return int(count) <= self.max_requests


def build_rate_limiter(
    backend: str = "memory",
    redis_url: str = "",
    max_requests: int = 60,
    window_seconds: int = 60,
):
    backend = backend.lower()
    if backend == "memory":
        return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)
    if backend == "redis":
        if not redis_url:
            raise ValueError("redis_url is required for Redis rate limiter")
        return RedisRateLimiter(redis_url, max_requests=max_requests, window_seconds=window_seconds)
    raise ValueError(f"Unsupported rate limiter backend: {backend}")
