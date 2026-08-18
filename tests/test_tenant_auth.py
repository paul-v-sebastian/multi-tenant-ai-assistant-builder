"""Unit tests for Phase 3 tenant auth helpers in app.py."""
from __future__ import annotations

from unittest.mock import patch

import bcrypt
import pytest


# ---------------------------------------------------------------------------
# _load_tenant_into_session
# ---------------------------------------------------------------------------

def test_load_tenant_into_session_sets_all_state_keys(monkeypatch):
    import app

    fake_state = {
        "tenant_id": None,
        "tenant_name": None,
        "tenant_status": None,
        "tenant_authenticated": False,
        "tenant_share_url": None,
        "cfg_index_name": "my-pdf-index",
        "cfg_top_k": 3,
        "cfg_min_confidence": 0.80,
    }
    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)

    row = {
        "id": "tenant-123",
        "name": "acme",
        "status": "INGESTED",
        "share_token": "tok-abc",
        "config": {"index_name": "custom-index", "top_k": 7, "min_confidence": 0.65},
    }
    app._load_tenant_into_session(row)

    assert fake_state["tenant_id"] == "tenant-123"
    assert fake_state["tenant_name"] == "acme"
    assert fake_state["tenant_status"] == "INGESTED"
    assert fake_state["tenant_authenticated"] is True
    assert fake_state["tenant_share_url"] == "tok-abc"
    assert fake_state["cfg_index_name"] == "custom-index"
    assert fake_state["cfg_top_k"] == 7
    assert fake_state["cfg_min_confidence"] == 0.65


def test_load_tenant_into_session_does_not_overwrite_defaults_when_config_empty(monkeypatch):
    import app

    fake_state = {
        "tenant_id": None,
        "tenant_name": None,
        "tenant_status": None,
        "tenant_authenticated": False,
        "tenant_share_url": None,
        "cfg_index_name": "my-pdf-index",
        "cfg_top_k": 3,
        "cfg_min_confidence": 0.80,
    }
    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)

    row = {"id": "t1", "name": "acme", "status": "DRAFT", "config": {}}
    app._load_tenant_into_session(row)

    # Config keys untouched when not present in stored config
    assert fake_state["cfg_index_name"] == "my-pdf-index"
    assert fake_state["cfg_top_k"] == 3
    assert fake_state["cfg_min_confidence"] == 0.80


# ---------------------------------------------------------------------------
# bcrypt hash / verify round-trip (core of the auth flow)
# ---------------------------------------------------------------------------

def test_bcrypt_roundtrip_correct_passkey():
    raw = "my-s3cret-passkey"
    hashed = bcrypt.hashpw(raw.encode(), bcrypt.gensalt()).decode()
    assert bcrypt.checkpw(raw.encode(), hashed.encode()) is True


def test_bcrypt_roundtrip_wrong_passkey():
    hashed = bcrypt.hashpw(b"correct", bcrypt.gensalt()).decode()
    assert bcrypt.checkpw(b"wrong", hashed.encode()) is False


# ---------------------------------------------------------------------------
# initialize_state includes tenant_name
# ---------------------------------------------------------------------------

def test_initialize_state_includes_tenant_name(monkeypatch):
    import app

    fake_state = {}
    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
    app.initialize_state()

    assert "tenant_name" in fake_state
    assert fake_state["tenant_name"] is None
