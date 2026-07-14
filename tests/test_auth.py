import pytest

from src.rag.auth import authenticate_api_key, parse_api_keys


def test_parse_api_keys_derives_roles_and_tenant():
    contexts = parse_api_keys("public-key:public,admin-key:public|admin:tenant-a")

    assert contexts["public-key"].roles == {"public"}
    assert contexts["admin-key"].roles == {"public", "admin"}
    assert contexts["admin-key"].tenant_id == "tenant-a"
    assert contexts["admin-key"].cache_scope() == "tenant-a:api-key:admin-:admin|public"


def test_authenticate_api_key_uses_dev_public_when_unconfigured():
    context = authenticate_api_key(None, {})

    assert context.subject == "dev-public"
    assert context.roles == {"public"}


def test_authenticate_api_key_rejects_missing_or_invalid_configured_key():
    contexts = parse_api_keys("public-key:public")

    with pytest.raises(PermissionError, match="missing API key"):
        authenticate_api_key(None, contexts)
    with pytest.raises(PermissionError, match="invalid API key"):
        authenticate_api_key("bad-key", contexts)
