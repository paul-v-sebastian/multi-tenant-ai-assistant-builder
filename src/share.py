"""Shareable URL generation and verification using HMAC-SHA256 signed JWTs.

POC spec:
  - Algorithm: HMAC-SHA256 (HS256)
  - Claim: ``tenant_id``
  - No expiry (``exp`` claim omitted for the POC)
  - URL format: ``https://<host>/?token=<jwt>``
"""
from __future__ import annotations

import jwt  # PyJWT


class ShareTokenError(Exception):
    """Raised when a share token cannot be generated or verified."""


def generate_share_token(
    tenant_id: str,
    secret: str,
    base_url: str = "",
) -> str:
    """Create an HMAC-SHA256 signed JWT and embed it in a shareable URL.

    Parameters
    ----------
    tenant_id:
        The UUID of the tenant to embed as the ``tenant_id`` claim.
    secret:
        The ``JWT_SECRET`` value from the environment.  Must not be empty.
    base_url:
        The application's public base URL (e.g. ``https://myapp.streamlit.app``).
        If empty the token is returned as a bare URL path: ``/?token=<jwt>``.

    Returns
    -------
    str
        Full shareable URL, e.g. ``https://myapp.streamlit.app/?token=<jwt>``.

    Raises
    ------
    ShareTokenError
        If *secret* is empty or JWT encoding fails.
    """
    if not secret:
        raise ShareTokenError("JWT_SECRET must not be empty.")
    if not tenant_id:
        raise ShareTokenError("tenant_id must not be empty.")

    try:
        token = jwt.encode({"tenant_id": tenant_id}, secret, algorithm="HS256")
    except Exception as exc:  # noqa: BLE001
        raise ShareTokenError(f"JWT encoding failed: {exc}") from exc

    base = base_url.rstrip("/") if base_url else ""
    return f"{base}/?token={token}"


def verify_share_token(token: str, secret: str) -> str:
    """Decode and verify a signed JWT, returning the ``tenant_id`` claim.

    Parameters
    ----------
    token:
        The raw JWT string (not the full URL — strip ``/?token=`` first).
    secret:
        The ``JWT_SECRET`` value.

    Returns
    -------
    str
        The ``tenant_id`` embedded in the token.

    Raises
    ------
    ShareTokenError
        If the token is invalid, tampered, or the ``tenant_id`` claim is absent.
    """
    if not secret:
        raise ShareTokenError("JWT_SECRET must not be empty.")
    if not token:
        raise ShareTokenError("Token must not be empty.")

    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.InvalidTokenError as exc:
        raise ShareTokenError(f"Invalid share token: {exc}") from exc

    tenant_id = payload.get("tenant_id")
    if not tenant_id:
        raise ShareTokenError("Token is missing the 'tenant_id' claim.")
    return str(tenant_id)
