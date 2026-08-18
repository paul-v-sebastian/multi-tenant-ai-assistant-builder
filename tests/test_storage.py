"""Unit tests for src/storage.py — all Supabase I/O is mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_storage_mock(signed_url: str = "https://example.com/signed"):
    """Return a MagicMock mimicking supabase storage fluent interface."""
    bucket = MagicMock()
    bucket.upload.return_value = {}
    bucket.create_signed_url.return_value = {"signedURL": signed_url}

    storage = MagicMock()
    storage.from_.return_value = bucket
    return storage, bucket


# ---------------------------------------------------------------------------
# upload_pdf
# ---------------------------------------------------------------------------

def test_upload_pdf_returns_signed_url():
    from src import storage

    storage_mock, bucket = _make_storage_mock("https://sb.io/pdfs/signed")
    mock_client = MagicMock()
    mock_client.storage = storage_mock

    with patch("src.supabase_client.get_client", return_value=mock_client):
        url = storage.upload_pdf("tenant-1", "doc.pdf", b"%PDF-content")

    assert url == "https://sb.io/pdfs/signed"
    bucket.upload.assert_called_once()
    call_kwargs = bucket.upload.call_args
    assert call_kwargs.kwargs["path"] == "tenant-1/doc.pdf"
    assert call_kwargs.kwargs["file"] == b"%PDF-content"


def test_upload_pdf_returns_none_when_not_connected():
    from src import storage

    with patch("src.supabase_client.get_client", return_value=None):
        url = storage.upload_pdf("tenant-1", "doc.pdf", b"%PDF")

    assert url is None


def test_upload_pdf_uses_upsert():
    from src import storage

    storage_mock, bucket = _make_storage_mock()
    mock_client = MagicMock()
    mock_client.storage = storage_mock

    with patch("src.supabase_client.get_client", return_value=mock_client):
        storage.upload_pdf("tenant-1", "doc.pdf", b"data")

    file_options = bucket.upload.call_args.kwargs["file_options"]
    assert file_options.get("upsert") == "true"


# ---------------------------------------------------------------------------
# upload_eval_csv
# ---------------------------------------------------------------------------

def test_upload_eval_csv_returns_signed_url():
    from src import storage

    storage_mock, bucket = _make_storage_mock("https://sb.io/eval-csvs/signed")
    mock_client = MagicMock()
    mock_client.storage = storage_mock

    with patch("src.supabase_client.get_client", return_value=mock_client):
        url = storage.upload_eval_csv("tenant-1", "gt.csv", b"query,expected\n")

    assert url == "https://sb.io/eval-csvs/signed"
    bucket.upload.assert_called_once()
    assert bucket.upload.call_args.kwargs["path"] == "tenant-1/gt.csv"


def test_upload_eval_csv_returns_none_when_not_connected():
    from src import storage

    with patch("src.supabase_client.get_client", return_value=None):
        url = storage.upload_eval_csv("tenant-1", "gt.csv", b"data")

    assert url is None


# ---------------------------------------------------------------------------
# get_pdf_url
# ---------------------------------------------------------------------------

def test_get_pdf_url_returns_signed_url():
    from src import storage

    storage_mock, bucket = _make_storage_mock("https://sb.io/pdfs/fresh-signed")
    mock_client = MagicMock()
    mock_client.storage = storage_mock

    with patch("src.supabase_client.get_client", return_value=mock_client):
        url = storage.get_pdf_url("tenant-1", "doc.pdf")

    assert url == "https://sb.io/pdfs/fresh-signed"


def test_get_pdf_url_returns_none_when_not_connected():
    from src import storage

    with patch("src.supabase_client.get_client", return_value=None):
        url = storage.get_pdf_url("tenant-1", "doc.pdf")

    assert url is None


def test_get_pdf_url_returns_none_on_storage_error():
    from src import storage

    storage_mock = MagicMock()
    storage_mock.from_.return_value.create_signed_url.side_effect = RuntimeError("not found")
    mock_client = MagicMock()
    mock_client.storage = storage_mock

    with patch("src.supabase_client.get_client", return_value=mock_client):
        url = storage.get_pdf_url("tenant-1", "missing.pdf")

    assert url is None


# ---------------------------------------------------------------------------
# Object path helper
# ---------------------------------------------------------------------------

def test_object_path_format():
    from src.storage import _object_path

    assert _object_path("tenant-abc", "report.pdf") == "tenant-abc/report.pdf"
