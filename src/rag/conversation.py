from __future__ import annotations

import re
from dataclasses import dataclass, field


FOLLOW_UP_PATTERN = re.compile(
    r"\b("
    r"it|its|that|this|they|them|those|he|she|his|her|their|same|above|previous|earlier|"
    r"more|second|third|first|last|point|detail|details"
    r")\b",
    re.I,
)


@dataclass(frozen=True)
class ConversationTurn:
    user: str
    assistant: str
    citations: list[str] = field(default_factory=list)


@dataclass
class ConversationMemoryStore:
    max_turns: int = 6
    values: dict[str, list[ConversationTurn]] = field(default_factory=dict)

    def key(self, session_id: str | None, workspace_id: str | None, auth_scope: str) -> str:
        return f"{workspace_id or 'default'}|{session_id or 'default'}|{auth_scope}"

    def get(self, session_id: str | None, workspace_id: str | None, auth_scope: str) -> list[ConversationTurn]:
        return list(self.values.get(self.key(session_id, workspace_id, auth_scope), []))

    def append(
        self,
        session_id: str | None,
        workspace_id: str | None,
        auth_scope: str,
        turn: ConversationTurn,
    ) -> None:
        key = self.key(session_id, workspace_id, auth_scope)
        turns = [*self.values.get(key, []), turn]
        self.values[key] = turns[-self.max_turns :]

    def clear(self) -> None:
        self.values.clear()


def build_contextual_query(query: str, turns: list[ConversationTurn], max_turns: int = 3) -> str:
    clean_query = " ".join(query.split())
    if not turns or not FOLLOW_UP_PATTERN.search(clean_query):
        return clean_query

    recent_user_questions = [" ".join(turn.user.split()) for turn in turns[-max_turns:] if turn.user.strip()]
    if not recent_user_questions:
        return clean_query
    history = " ".join(recent_user_questions)
    return f"Conversation context: {history}. Follow-up question: {clean_query}"
