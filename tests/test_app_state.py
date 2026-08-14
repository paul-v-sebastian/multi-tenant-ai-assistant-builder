import types

from app import _GLOBAL_CSS, is_chat_ready


def test_is_chat_ready_requires_loaded_document():
    assert is_chat_ready({"namespace": "ns-123", "document_name": "doc.pdf", "chunk_count": 4}) is True
    assert is_chat_ready({"namespace": None, "document_name": "doc.pdf", "chunk_count": 4}) is False
    assert is_chat_ready({"namespace": "ns-123", "document_name": None, "chunk_count": 4}) is False
    assert is_chat_ready({"namespace": "ns-123", "document_name": "doc.pdf", "chunk_count": 0}) is False


def test_global_css_keeps_main_app_visible():
    assert '[data-testid="stAppViewContainer"] > section:first-child' not in _GLOBAL_CSS


def test_show_chat_error_renders_explicit_ui_error(monkeypatch):
    import app

    fake_state = {"messages": []}
    captured = {}

    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
    monkeypatch.setattr(app.st, "error", lambda message: captured.setdefault("error", message), raising=False)

    app.show_chat_error(RuntimeError("openai failed"))

    assert captured["error"] == "⚠️ An error occurred: openai failed"
    assert fake_state["messages"][-1]["role"] == "assistant"
    assert fake_state["messages"][-1]["content"] == "⚠️ An error occurred: openai failed"
