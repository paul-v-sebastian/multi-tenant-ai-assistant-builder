import app


def test_global_css_exists():
    assert "<style>" in app._GLOBAL_CSS


def test_clear_conversation(monkeypatch):
    fake_state = {"messages": [{"role": "user", "content": "hi"}]}
    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)

    app.clear_conversation()

    assert fake_state["messages"] == []


def test_show_chat_error_renders_explicit_ui_error(monkeypatch):
    fake_state = {"messages": []}
    captured = {}

    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
    monkeypatch.setattr(app.st, "error", lambda message: captured.setdefault("error", message), raising=False)

    app.show_chat_error(RuntimeError("openai failed"))

    assert captured["error"] == "⚠️ An error occurred: openai failed"
    assert fake_state["messages"][-1]["role"] == "assistant"
    assert fake_state["messages"][-1]["content"] == "⚠️ An error occurred: openai failed"
