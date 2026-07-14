from __future__ import annotations

import re
import time
from collections import defaultdict, deque


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
