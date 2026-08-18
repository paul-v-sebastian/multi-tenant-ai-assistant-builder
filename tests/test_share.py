"""Unit tests for src/share.py — HMAC-SHA256 signed JWT share tokens."""
from __future__ import annotations

import pytest

from src.share import ShareTokenError, generate_share_token, verify_share_token

_SECRET = "test-jwt-secret-value"
_TENANT = "tenant-abc-123"


# ---------------------------------------------------------------------------
# generate_share_token
# ---------------------------------------------------------------------------

def test_generate_returns_url_containing_token():
    url = generate_share_token(_TENANT, _SECRET, base_url="https://myapp.example.com")
    assert url.startswith("https://myapp.example.com/?token=")
    # JWT has three dot-separated segments
    token_part = url.split("?token=")[1]
    assert token_part.count(".") == 2


def test_generate_without_base_url_returns_relative_path():
    url = generate_share_token(_TENANT, _SECRET, base_url="")
    assert url.startswith("/?token=")


def test_generate_raises_on_empty_secret():
    with pytest.raises(ShareTokenError, match="JWT_SECRET"):
        generate_share_token(_TENANT, "", base_url="")


def test_generate_raises_on_empty_tenant_id():
    with pytest.raises(ShareTokenError, match="tenant_id"):
        generate_share_token("", _SECRET, base_url="")


# ---------------------------------------------------------------------------
# verify_share_token
# ---------------------------------------------------------------------------

def test_roundtrip_verify_returns_tenant_id():
    url = generate_share_token(_TENANT, _SECRET, base_url="https://myapp.example.com")
    token = url.split("?token=")[1]
    result = verify_share_token(token, _SECRET)
    assert result == _TENANT


def test_verify_rejects_wrong_secret():
    url = generate_share_token(_TENANT, _SECRET)
    token = url.split("?token=")[1]
    with pytest.raises(ShareTokenError, match="Invalid share token"):
        verify_share_token(token, "wrong-secret")


def test_verify_rejects_tampered_token():
    url = generate_share_token(_TENANT, _SECRET)
    token = url.split("?token=")[1]
    # Flip one character in the signature part
    tampered = token[:-1] + ("X" if token[-1] != "X" else "Y")
    with pytest.raises(ShareTokenError):
        verify_share_token(tampered, _SECRET)


def test_verify_raises_on_empty_token():
    with pytest.raises(ShareTokenError, match="Token must not be empty"):
        verify_share_token("", _SECRET)


def test_verify_raises_on_empty_secret():
    with pytest.raises(ShareTokenError, match="JWT_SECRET"):
        verify_share_token("some.token.here", "")


def test_verify_raises_when_tenant_id_claim_missing():
    import jwt  # noqa: PLC0415

    # Token without tenant_id claim
    token = jwt.encode({"sub": "someone"}, _SECRET, algorithm="HS256")
    with pytest.raises(ShareTokenError, match="missing the 'tenant_id' claim"):
        verify_share_token(token, _SECRET)
