import pytest

from src.rag.auth import authenticate_jwt, authenticate_request, parse_api_keys, sign_jwt


def test_parse_api_keys_derives_roles_and_tenant():
    contexts = parse_api_keys("public-key:public,admin-key:public|admin:tenant-a")

    assert contexts["public-key"].roles == {"public"}
    assert contexts["admin-key"].roles == {"public", "admin"}
    assert contexts["admin-key"].tenant_id == "tenant-a"
    assert contexts["admin-key"].cache_scope() == "tenant-a:api-key:admin-:admin|public"


def test_authenticate_api_key_uses_dev_public_when_unconfigured():
    context = authenticate_request("auto", None, None, {})

    assert context.subject == "dev-public"
    assert context.roles == {"public"}


def test_authenticate_api_key_rejects_missing_or_invalid_configured_key():
    contexts = parse_api_keys("public-key:public")

    with pytest.raises(PermissionError, match="missing API key"):
        authenticate_request("api_key", None, None, contexts)
    with pytest.raises(PermissionError, match="invalid API key"):
        authenticate_request("api_key", "bad-key", None, contexts)


def test_authenticate_jwt_derives_subject_roles_and_tenant():
    token = sign_jwt(
        {
            "sub": "user-1",
            "roles": ["public", "admin"],
            "tenant_id": "tenant-a",
            "iss": "issuer",
            "aud": "rag-api",
            "exp": 2_000_000_000,
        },
        "secret",
    )

    context = authenticate_jwt(f"Bearer {token}", "secret", issuer="issuer", audience="rag-api", now=1_700_000_000)

    assert context.subject == "jwt:user-1"
    assert context.roles == {"public", "admin"}
    assert context.tenant_id == "tenant-a"


def test_authenticate_jwt_rejects_invalid_signature_and_expired_token():
    valid = sign_jwt({"sub": "user-1", "exp": 2_000_000_000}, "secret")
    expired = sign_jwt({"sub": "user-1", "exp": 1}, "secret")

    with pytest.raises(PermissionError, match="invalid JWT signature"):
        authenticate_jwt(f"Bearer {valid}", "wrong-secret", now=1_700_000_000)
    with pytest.raises(PermissionError, match="expired bearer token"):
        authenticate_jwt(f"Bearer {expired}", "secret", now=1_700_000_000)
