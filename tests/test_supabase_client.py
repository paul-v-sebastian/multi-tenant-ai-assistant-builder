"""Unit tests for src/supabase_client.py — all Supabase I/O is mocked."""
from __future__ import annotations

import importlib
import sys
import uuid
from unittest.mock import MagicMock, patch

import bcrypt
import pytest


# ---------------------------------------------------------------------------
# Helpers to reset the module-level singleton between tests
# ---------------------------------------------------------------------------

def _reset_module():
    """Force a fresh import of supabase_client so singleton state is clean."""
    mod_name = "src.supabase_client"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


def _make_mock_client():
    """Return a MagicMock that mimics a supabase.Client fluent interface."""
    mock = MagicMock()
    # table().insert().execute()  / table().select()…execute() etc.
    mock.table.return_value = mock
    mock.insert.return_value = mock
    mock.select.return_value = mock
    mock.update.return_value = mock
    mock.eq.return_value = mock
    mock.limit.return_value = mock
    mock.execute.return_value = MagicMock(data=[])
    # storage.list_buckets() used by init_supabase
    mock.storage.list_buckets.return_value = []
    return mock


# ---------------------------------------------------------------------------
# init_supabase / get_supabase_status
# ---------------------------------------------------------------------------

def test_init_supabase_grey_when_no_credentials():
    sc = _reset_module()
    sc.init_supabase("", "")
    status, msg = sc.get_supabase_status()
    assert status == "grey"
    assert "not configured" in msg


def test_init_supabase_green_on_success():
    sc = _reset_module()
    mock_client = _make_mock_client()
    with patch("supabase.create_client", return_value=mock_client):
        sc.init_supabase("https://example.supabase.co", "service-key")
    status, msg = sc.get_supabase_status()
    assert status == "green"
    assert sc.get_client() is mock_client


def test_init_supabase_red_on_exception():
    sc = _reset_module()
    with patch("supabase.create_client", side_effect=RuntimeError("timeout")):
        sc.init_supabase("https://example.supabase.co", "service-key")
    status, msg = sc.get_supabase_status()
    assert status == "red"
    assert "timeout" in msg


# ---------------------------------------------------------------------------
# create_tenant
# ---------------------------------------------------------------------------

def test_create_tenant_raises_when_not_connected():
    sc = _reset_module()
    with pytest.raises(sc.SupabaseError, match="not connected"):
        sc.create_tenant("acme", "hash")


def test_create_tenant_returns_uuid_and_inserts_draft():
    sc = _reset_module()
    mock_client = _make_mock_client()
    sc._client = mock_client  # inject connected client

    tenant_id = sc.create_tenant("acme", "bcrypt-hash")

    assert uuid.UUID(tenant_id)  # valid UUID
    # Verify table('tenants').insert(...).execute() was called
    mock_client.table.assert_called_with("tenants")
    insert_call_kwargs = mock_client.insert.call_args[0][0]
    assert insert_call_kwargs["name"] == "acme"
    assert insert_call_kwargs["status"] == "DRAFT"
    assert insert_call_kwargs["passkey_hash"] == "bcrypt-hash"


# ---------------------------------------------------------------------------
# get_tenant
# ---------------------------------------------------------------------------

def test_get_tenant_returns_none_when_not_found():
    sc = _reset_module()
    mock_client = _make_mock_client()
    mock_client.execute.return_value = MagicMock(data=[])
    sc._client = mock_client

    result = sc.get_tenant("non-existent-id")
    assert result is None


def test_get_tenant_returns_row_dict():
    sc = _reset_module()
    mock_client = _make_mock_client()
    row = {"id": "abc-123", "name": "acme", "status": "DRAFT", "passkey_hash": "h"}
    mock_client.execute.return_value = MagicMock(data=[row])
    sc._client = mock_client

    result = sc.get_tenant("abc-123")
    assert result == row


# ---------------------------------------------------------------------------
# get_tenant_by_name  (Bug 1 fix)
# ---------------------------------------------------------------------------

def test_get_tenant_by_name_returns_none_when_not_found():
    sc = _reset_module()
    mock_client = _make_mock_client()
    mock_client.execute.return_value = MagicMock(data=[])
    sc._client = mock_client

    result = sc.get_tenant_by_name("nonexistent")
    assert result is None


def test_get_tenant_by_name_returns_row_dict():
    sc = _reset_module()
    mock_client = _make_mock_client()
    row = {"id": "abc-123", "name": "acme", "status": "DRAFT", "passkey_hash": "h"}
    mock_client.execute.return_value = MagicMock(data=[row])
    sc._client = mock_client

    result = sc.get_tenant_by_name("acme")
    assert result == row
    mock_client.eq.assert_called_with("name", "acme")


def test_get_tenant_by_name_raises_when_not_connected():
    sc = _reset_module()
    with pytest.raises(sc.SupabaseError, match="not connected"):
        sc.get_tenant_by_name("acme")


# ---------------------------------------------------------------------------
# verify_tenant_passkey
# ---------------------------------------------------------------------------

def test_verify_tenant_passkey_returns_true_for_correct_passkey():
    sc = _reset_module()
    mock_client = _make_mock_client()
    raw = "s3cret"
    hashed = bcrypt.hashpw(raw.encode(), bcrypt.gensalt()).decode()
    row = {"id": "t1", "name": "acme", "status": "DRAFT", "passkey_hash": hashed}
    mock_client.execute.return_value = MagicMock(data=[row])
    sc._client = mock_client

    assert sc.verify_tenant_passkey("t1", raw) is True


def test_verify_tenant_passkey_returns_false_for_wrong_passkey():
    sc = _reset_module()
    mock_client = _make_mock_client()
    hashed = bcrypt.hashpw(b"correct", bcrypt.gensalt()).decode()
    row = {"id": "t1", "name": "acme", "status": "DRAFT", "passkey_hash": hashed}
    mock_client.execute.return_value = MagicMock(data=[row])
    sc._client = mock_client

    assert sc.verify_tenant_passkey("t1", "wrong") is False


def test_verify_tenant_passkey_raises_for_missing_tenant():
    sc = _reset_module()
    mock_client = _make_mock_client()
    mock_client.execute.return_value = MagicMock(data=[])
    sc._client = mock_client

    with pytest.raises(sc.SupabaseError, match="not found"):
        sc.verify_tenant_passkey("ghost", "pass")


# ---------------------------------------------------------------------------
# update_tenant_status
# ---------------------------------------------------------------------------

def test_update_tenant_status_happy_path():
    sc = _reset_module()
    mock_client = _make_mock_client()
    row = {"id": "t1", "name": "acme", "status": "DRAFT", "passkey_hash": "h"}
    mock_client.execute.return_value = MagicMock(data=[row])
    sc._client = mock_client

    sc.update_tenant_status("t1", "INGESTED")
    mock_client.update.assert_called_with({"status": "INGESTED"})


def test_update_tenant_status_rejects_invalid_transition():
    sc = _reset_module()
    mock_client = _make_mock_client()
    row = {"id": "t1", "name": "acme", "status": "DRAFT", "passkey_hash": "h"}
    mock_client.execute.return_value = MagicMock(data=[row])
    sc._client = mock_client

    with pytest.raises(sc.SupabaseError, match="Invalid status transition"):
        sc.update_tenant_status("t1", "PUBLISHED")


# ---------------------------------------------------------------------------
# upsert_tenant_config
# ---------------------------------------------------------------------------

def test_upsert_tenant_config_calls_update():
    sc = _reset_module()
    mock_client = _make_mock_client()
    sc._client = mock_client

    cfg = {"top_k": 5, "min_confidence": 0.75}
    sc.upsert_tenant_config("t1", cfg)

    mock_client.update.assert_called_with({"config": cfg})


# ---------------------------------------------------------------------------
# save_eval_log
# ---------------------------------------------------------------------------

def test_save_eval_log_inserts_all_rows():
    sc = _reset_module()
    mock_client = _make_mock_client()
    sc._client = mock_client

    rows = [
        {"query": "q1", "expected": "e1", "actual": "a1", "score": 4, "reason": "good"},
        {"query": "q2", "expected": "e2", "actual": "a2", "score": 2, "reason": "bad"},
    ]
    sc.save_eval_log("tenant-abc", rows)

    inserted = mock_client.insert.call_args[0][0]
    assert len(inserted) == 2
    assert inserted[0]["tenant_id"] == "tenant-abc"
    assert inserted[0]["query"] == "q1"
    assert inserted[1]["score"] == 2


def test_save_eval_log_raises_when_not_connected():
    sc = _reset_module()
    with pytest.raises(sc.SupabaseError, match="not connected"):
        sc.save_eval_log("t1", [])
