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


def test_initialize_state_sets_index_keys(monkeypatch):
    fake_state = {}
    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)

    app.initialize_state()

    assert fake_state["index_built"] is False
    assert fake_state["vector_namespace"] is None


def test_build_vector_store_passes_config(monkeypatch):
    from src.config import AppConfig

    config = AppConfig(
        openai_api_key="sk-test",
        pinecone_api_key="pc-test",
        pinecone_index_name="test-index",
        pinecone_cloud="aws",
        pinecone_region="us-east-1",
    )

    created = {}

    class FakeVectorStore:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(app, "PineconeVectorStore", FakeVectorStore)

    app.build_vector_store(config)

    assert created["api_key"] == "pc-test"
    assert created["index_name"] == "test-index"
    assert created["dimension"] == config.embedding_dimension
