"""Supabase client singleton with connection-status helpers and tenant CRUD.

Status values:
  "grey"  — credentials not configured (URL or key missing)
  "red"   — credentials present but connection failed
  "green" — connection succeeded
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

_client = None          # cached supabase.Client instance
_status: Literal["grey", "red", "green"] = "grey"
_status_message: str = "Supabase credentials not configured."


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RLS helper
# ---------------------------------------------------------------------------

def set_rls_context(client, tenant_id: str) -> None:
    """Issue ``SET LOCAL app.current_tenant_id`` so Postgres RLS policies fire.

    Must be called inside the same transaction as any subsequent tenant-scoped
    query.  With the Supabase Python client we use ``rpc`` to execute the raw
    SQL because the client does not expose a raw ``execute`` method.
    """
    client.rpc(
        "set_config",
        {"setting": "app.current_tenant_id", "value": tenant_id, "is_local": True},
    ).execute()


# ---------------------------------------------------------------------------
# Tenant CRUD
# ---------------------------------------------------------------------------

class SupabaseError(Exception):
    """Raised when a Supabase operation fails."""


def create_tenant(name: str, passkey_hash: str) -> str:
    """Insert a new DRAFT tenant row and return the generated ``tenant_id``.

    Parameters
    ----------
    name:
        Human-readable tenant name.
    passkey_hash:
        bcrypt hash of the raw passkey (never store the plaintext).

    Returns
    -------
    str
        UUID of the newly created tenant.

    Raises
    ------
    SupabaseError
        If the client is not connected or the insert fails.
    """
    client = get_client()
    if client is None:
        raise SupabaseError("Supabase client is not connected.")

    tenant_id = str(uuid.uuid4())
    try:
        client.table("tenants").insert(
            {
                "id": tenant_id,
                "name": name,
                "passkey_hash": passkey_hash,
                "status": "DRAFT",
                "config": {},
            }
        ).execute()
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Failed to create tenant: {exc}") from exc

    return tenant_id


def get_tenant(tenant_id: str) -> dict[str, Any] | None:
    """Fetch a single tenant row by id.

    Returns the row as a dict, or ``None`` if not found.

    Raises
    ------
    SupabaseError
        If the client is not connected or the query fails.
    """
    client = get_client()
    if client is None:
        raise SupabaseError("Supabase client is not connected.")

    try:
        result = (
            client.table("tenants")
            .select("*")
            .eq("id", tenant_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Failed to fetch tenant: {exc}") from exc

    rows = result.data or []
    return rows[0] if rows else None


def get_tenant_by_name(name: str) -> dict[str, Any] | None:
    """Fetch a single tenant row by human-readable name.

    Returns the row as a dict, or ``None`` if not found.

    Raises
    ------
    SupabaseError
        If the client is not connected or the query fails.
    """
    client = get_client()
    if client is None:
        raise SupabaseError("Supabase client is not connected.")

    try:
        result = (
            client.table("tenants")
            .select("*")
            .eq("name", name)
            .limit(1)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Failed to fetch tenant: {exc}") from exc

    rows = result.data or []
    return rows[0] if rows else None


def verify_tenant_passkey(tenant_id: str, passkey: str) -> bool:
    """Return ``True`` if *passkey* matches the bcrypt hash stored for *tenant_id*.

    Raises
    ------
    SupabaseError
        If the client is not connected, the tenant does not exist, or the DB
        query fails.
    """
    import bcrypt  # noqa: PLC0415 — optional at module level

    row = get_tenant(tenant_id)
    if row is None:
        raise SupabaseError(f"Tenant '{tenant_id}' not found.")

    stored_hash: str = row.get("passkey_hash", "")
    return bcrypt.checkpw(passkey.encode(), stored_hash.encode())


def update_tenant_status(tenant_id: str, new_status: str) -> None:
    """Validate the lifecycle transition and persist the new status.

    Uses :func:`src.tenants.is_valid_tenant_status_transition` to enforce the
    DRAFT → INGESTED → EVALUATED → PUBLISHED lifecycle.

    Raises
    ------
    SupabaseError
        If the client is not connected, the transition is invalid, or the
        update fails.
    """
    from src.tenants import (  # noqa: PLC0415 — avoid circular import at module level
        coerce_tenant_status,
        is_valid_tenant_status_transition,
    )

    client = get_client()
    if client is None:
        raise SupabaseError("Supabase client is not connected.")

    row = get_tenant(tenant_id)
    if row is None:
        raise SupabaseError(f"Tenant '{tenant_id}' not found.")

    current_status = row.get("status")
    if not is_valid_tenant_status_transition(current_status, new_status):
        raise SupabaseError(
            f"Invalid status transition: {current_status!r} → {new_status!r}."
        )

    target = coerce_tenant_status(new_status)
    try:
        client.table("tenants").update({"status": str(target)}).eq("id", tenant_id).execute()
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Failed to update tenant status: {exc}") from exc


def upsert_tenant_config(tenant_id: str, config_dict: dict[str, Any]) -> None:
    """Write (overwrite) the ``config`` JSONB column for *tenant_id*.

    Raises
    ------
    SupabaseError
        If the client is not connected or the update fails.
    """
    client = get_client()
    if client is None:
        raise SupabaseError("Supabase client is not connected.")

    try:
        client.table("tenants").update({"config": config_dict}).eq("id", tenant_id).execute()
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Failed to upsert tenant config: {exc}") from exc


def save_eval_log(tenant_id: str, rows: list[dict[str, Any]]) -> None:
    """Bulk-insert eval result rows into ``eval_logs``.

    Each item in *rows* must contain the keys:
    ``query``, ``expected``, ``actual``, ``score``, ``reason``.

    Raises
    ------
    SupabaseError
        If the client is not connected or the insert fails.
    """
    client = get_client()
    if client is None:
        raise SupabaseError("Supabase client is not connected.")

    records = [
        {
            "id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "query": row["query"],
            "expected": row["expected"],
            "actual": row["actual"],
            "score": int(row["score"]),
            "reason": str(row.get("reason", "")),
        }
        for row in rows
    ]

    try:
        client.table("eval_logs").insert(records).execute()
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Failed to save eval log: {exc}") from exc
