"""Supabase Storage bucket helpers for PDF and eval-CSV persistence.

Bucket layout (must be created in the Supabase dashboard or via migration):
  pdfs/       — one file per tenant PDF: ``<tenant_id>/<filename>``
  eval-csvs/  — one file per ground-truth CSV: ``<tenant_id>/<filename>``

All functions return ``None`` (and log a warning) when the Supabase client is
not connected, so the rest of the app continues to work in environments where
Supabase credentials are absent.
"""
from __future__ import annotations

_PDF_BUCKET = "pdfs"
_EVAL_BUCKET = "eval-csvs"
_SIGNED_URL_EXPIRES = 3600  # seconds (1 hour)


def _object_path(tenant_id: str, filename: str) -> str:
    """Return the storage object path ``<tenant_id>/<filename>``."""
    return f"{tenant_id}/{filename}"


def upload_pdf(tenant_id: str, filename: str, data: bytes) -> str | None:
    """Upload a PDF to the ``pdfs`` bucket and return a signed download URL.

    Parameters
    ----------
    tenant_id:
        The authenticated tenant's UUID.
    filename:
        Original filename (e.g. ``report.pdf``).
    data:
        Raw PDF bytes.

    Returns
    -------
    str | None
        A signed URL valid for :data:`_SIGNED_URL_EXPIRES` seconds, or
        ``None`` when the client is not connected.
    """
    from src.supabase_client import get_client  # noqa: PLC0415

    client = get_client()
    if client is None:
        return None

    path = _object_path(tenant_id, filename)
    client.storage.from_(_PDF_BUCKET).upload(
        path=path,
        file=data,
        file_options={"content-type": "application/pdf", "upsert": "true"},
    )
    result = client.storage.from_(_PDF_BUCKET).create_signed_url(
        path, _SIGNED_URL_EXPIRES
    )
    return result.get("signedURL") or result.get("signedUrl")


def upload_eval_csv(tenant_id: str, filename: str, data: bytes) -> str | None:
    """Upload a ground-truth CSV to the ``eval-csvs`` bucket and return a URL.

    Parameters
    ----------
    tenant_id:
        The authenticated tenant's UUID.
    filename:
        Original filename (e.g. ``ground_truth.csv``).
    data:
        Raw CSV bytes.

    Returns
    -------
    str | None
        A signed URL, or ``None`` when the client is not connected.
    """
    from src.supabase_client import get_client  # noqa: PLC0415

    client = get_client()
    if client is None:
        return None

    path = _object_path(tenant_id, filename)
    client.storage.from_(_EVAL_BUCKET).upload(
        path=path,
        file=data,
        file_options={"content-type": "text/csv", "upsert": "true"},
    )
    result = client.storage.from_(_EVAL_BUCKET).create_signed_url(
        path, _SIGNED_URL_EXPIRES
    )
    return result.get("signedURL") or result.get("signedUrl")


def get_pdf_url(tenant_id: str, filename: str) -> str | None:
    """Return a fresh signed URL for an already-uploaded PDF.

    Returns ``None`` when the client is not connected or the object does not
    exist.
    """
    from src.supabase_client import get_client  # noqa: PLC0415

    client = get_client()
    if client is None:
        return None

    path = _object_path(tenant_id, filename)
    try:
        result = client.storage.from_(_PDF_BUCKET).create_signed_url(
            path, _SIGNED_URL_EXPIRES
        )
        return result.get("signedURL") or result.get("signedUrl")
    except Exception:  # noqa: BLE001
        return None
