from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
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


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def sign_jwt(payload: dict, secret: str, header: dict | None = None) -> str:
    header = header or {"alg": "HS256", "typ": "JWT"}
    encoded_header = _b64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    encoded_payload = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{encoded_header}.{encoded_payload}.{_b64url_encode(signature)}"


def authenticate_jwt(
    authorization_header: str | None,
    secret: str,
    issuer: str = "",
    audience: str = "",
    now: int | None = None,
) -> AuthContext:
    if not secret:
        raise PermissionError("JWT auth is not configured")
    if not authorization_header or not authorization_header.startswith("Bearer "):
        raise PermissionError("missing bearer token")
    token = authorization_header.removeprefix("Bearer ").strip()
    parts = token.split(".")
    if len(parts) != 3:
        raise PermissionError("invalid bearer token")

    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    try:
        signature = _b64url_decode(parts[2])
        header = json.loads(_b64url_decode(parts[0]))
        payload = json.loads(_b64url_decode(parts[1]))
    except Exception as exc:
        raise PermissionError("invalid bearer token") from exc

    if header.get("alg") != "HS256":
        raise PermissionError("unsupported JWT algorithm")
    if not hmac.compare_digest(signature, expected):
        raise PermissionError("invalid JWT signature")

    now = int(time.time()) if now is None else now
    if "exp" in payload and int(payload["exp"]) < now:
        raise PermissionError("expired bearer token")
    if issuer and payload.get("iss") != issuer:
        raise PermissionError("invalid JWT issuer")
    if audience:
        claim = payload.get("aud")
        audiences = claim if isinstance(claim, list) else [claim]
        if audience not in audiences:
            raise PermissionError("invalid JWT audience")

    subject = str(payload.get("sub") or "")
    if not subject:
        raise PermissionError("missing JWT subject")
    raw_roles = payload.get("roles", ["public"])
    if isinstance(raw_roles, str):
        raw_roles = [raw_roles]
    roles = {str(role) for role in raw_roles if str(role)}
    if not roles:
        roles = {"public"}
    tenant_id = str(payload.get("tenant_id") or payload.get("tenant") or "default")
    return AuthContext(subject=f"jwt:{subject}", roles=roles, tenant_id=tenant_id)


def authenticate_request(
    auth_mode: str,
    api_key: str | None,
    authorization_header: str | None,
    configured_keys: dict[str, AuthContext],
    jwt_secret: str = "",
    jwt_issuer: str = "",
    jwt_audience: str = "",
) -> AuthContext:
    mode = auth_mode.lower()
    if mode == "auto":
        if configured_keys:
            return authenticate_api_key(api_key, configured_keys, allow_dev_public=False)
        return AuthContext(subject="dev-public", roles={"public"})
    if mode == "dev":
        return AuthContext(subject="dev-public", roles={"public"})
    if mode == "api_key":
        return authenticate_api_key(api_key, configured_keys, allow_dev_public=False)
    if mode == "jwt":
        return authenticate_jwt(
            authorization_header,
            secret=jwt_secret,
            issuer=jwt_issuer,
            audience=jwt_audience,
        )
    raise PermissionError(f"unsupported auth mode: {auth_mode}")
