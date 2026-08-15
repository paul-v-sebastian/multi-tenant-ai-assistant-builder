"""Langfuse tracing helpers for the RAG app.

The Langfuse client is initialised lazily on first call to ``get_langfuse()``.
If Langfuse credentials are absent the helper returns ``None`` and all callers
must guard with ``if lf:`` so the app degrades gracefully without tracing.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_langfuse_client = None
_langfuse_initialised = False


def init_langfuse(secret_key: str, public_key: str, host: str) -> None:
    """Initialise the module-level Langfuse client.

    Safe to call multiple times — subsequent calls are no-ops once the client
    has been successfully created.  If credentials are missing the client is
    set to ``None`` so callers receive a consistent ``None`` sentinel.
    """
    global _langfuse_client, _langfuse_initialised
    if _langfuse_initialised:
        return
    _langfuse_initialised = True
    if not (secret_key and public_key):
        logger.debug("Langfuse credentials not configured — tracing disabled.")
        return
    try:
        from langfuse import Langfuse  # local import keeps startup fast when absent

        _langfuse_client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )
        logger.debug("Langfuse client initialised (host=%s).", host)
    except Exception:  # pragma: no cover
        logger.warning("Failed to initialise Langfuse client.", exc_info=True)


def get_langfuse():
    """Return the module-level Langfuse client, or ``None`` if not configured."""
    return _langfuse_client
