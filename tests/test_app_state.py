import app
from src.config import AppConfig
from src.retrieval import RetrievedChunk


def test_global_css_exists():
    assert "<style>" in app._GLOBAL_CSS


def test_initialize_state_sets_defaults(monkeypatch):
    fake_state = {}
    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)

    app.initialize_state()

    assert fake_state["messages"] == []
    assert fake_state["show_sources_by_default"] is False


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


def test_build_message_serializes_sources():
    sources = [RetrievedChunk("1", "Chunk text", "doc.pdf", 2, 4, 0.87)]

    message = app.build_message("assistant", "Answer", sources=sources)

    assert message["sources"] == [
        {
            "chunk_id": "1",
            "text": "Chunk text",
            "source": "doc.pdf",
            "chunk_index": 2,
            "page": 4,
            "score": 0.87,
        }
    ]


def test_retrieve_relevant_chunks_returns_empty_without_pinecone_key():
    config = AppConfig(openai_api_key="test-key", pinecone_api_key="")

    assert app.retrieve_relevant_chunks("hello", config) == []
