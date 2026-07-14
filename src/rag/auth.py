from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthContext:
    subject: str
    roles: set[str]
    tenant_id: str = "default"

    def cache_scope(self) -> str:
        roles = "|".join(sorted(self.roles))
        return f"{self.tenant_id}:{self.subject}:{roles}"


def parse_api_keys(raw: str) -> dict[str, AuthContext]:
    contexts = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) not in {2, 3} or not parts[0] or not parts[1]:
            raise ValueError("API keys must use key:role1|role2 or key:role1|role2:tenant format")
        key, roles_raw = parts[0], parts[1]
        tenant_id = parts[2] if len(parts) == 3 and parts[2] else "default"
        roles = {role.strip() for role in roles_raw.split("|") if role.strip()}
        if not roles:
            raise ValueError("API key role list cannot be empty")
        contexts[key] = AuthContext(subject=f"api-key:{key[:6]}", roles=roles, tenant_id=tenant_id)
    return contexts


def authenticate_api_key(
    api_key: str | None,
    configured_keys: dict[str, AuthContext],
    allow_dev_public: bool = True,
) -> AuthContext:
    if not configured_keys and allow_dev_public:
        return AuthContext(subject="dev-public", roles={"public"})
    if not api_key:
        raise PermissionError("missing API key")
    context = configured_keys.get(api_key)
    if context is None:
        raise PermissionError("invalid API key")
    return context
