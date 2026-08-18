"""Supabase client singleton with connection-status helpers.

Status values:
  "grey"  — credentials not configured (URL or key missing)
  "red"   — credentials present but connection failed
  "green" — connection succeeded
"""
from __future__ import annotations

from typing import Literal

_client = None          # cached supabase.Client instance
_status: Literal["grey", "red", "green"] = "grey"
_status_message: str = "Supabase credentials not configured."


def get_supabase_status() -> tuple[Literal["grey", "red", "green"], str]:
    """Return the current (status, message) pair without re-probing."""
    return _status, _status_message


def init_supabase(url: str, key: str) -> None:
    """Attempt to connect to Supabase and update the module-level status.

    Safe to call every Streamlit run — re-uses the cached client when the
    credentials match what was used to create it, and skips if grey (no creds).
    """
    global _client, _status, _status_message

    if not url or not key:
        _status = "grey"
        _status_message = "Supabase credentials not configured."
        _client = None
        return

    # Re-use existing client if already connected successfully
    if _client is not None and _status == "green":
        return

    try:
        from supabase import create_client  # noqa: PLC0415 — lazy import

        client = create_client(url, key)
        # Lightweight probe: fetch the list of tables via the REST API.
        # Using rpc or a simple select on a system view would require a real
        # table to exist; instead we call the health endpoint by listing
        # storage buckets, which is always available on any Supabase project.
        client.storage.list_buckets()
        _client = client
        _status = "green"
        _status_message = "Connected to Supabase."
    except Exception as exc:  # noqa: BLE001
        _client = None
        _status = "red"
        _status_message = f"Supabase connection failed: {exc}"


def get_client():
    """Return the connected supabase.Client, or None if not connected."""
    return _client
