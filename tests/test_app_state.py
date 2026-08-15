import app


def test_global_css_exists():
    assert "<style>" in app._GLOBAL_CSS


def test_clear_conversation(monkeypatch):
    fake_state = {
        "messages": [{"role": "user", "content": "hi"}],
        "last_retrieval_debug": {"Threshold": "0.80"},
    }
    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)

    app.clear_conversation()

    assert fake_state["messages"] == []
    assert fake_state["last_retrieval_debug"] is None


def test_show_chat_error_renders_explicit_ui_error(monkeypatch):
    fake_state = {"messages": [{"role": "user", "content": "hello"}]}
    captured = {}

    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
    monkeypatch.setattr(app.st, "error", lambda message: captured.setdefault("error", message), raising=False)

    app.show_chat_error(RuntimeError("openai failed"))

    assert captured["error"] == "⚠️ An error occurred: openai failed"
    assert fake_state["messages"][-1]["role"] == "assistant"
    assert fake_state["messages"][-1]["content"] == "⚠️ An error occurred: openai failed"
    assert fake_state["messages"][0]["role"] == "assistant"


def test_initialize_state_sets_index_keys(monkeypatch):
    fake_state = {}
    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)

    app.initialize_state()

    assert fake_state["index_built"] is False
    assert fake_state["vector_namespace"] is None
    assert fake_state["pending_question"] is None
    assert fake_state["last_retrieval_debug"] is None


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


def test_parse_eval_csv_accepts_required_schema():
    rows = app.parse_eval_csv(
        b"Query,Expected Response\nWhat is RAG?,Retrieval augmented generation\n"
    )

    assert rows == [{"Query": "What is RAG?", "Expected Response": "Retrieval augmented generation"}]


def test_parse_eval_csv_rejects_missing_required_columns():
    try:
        app.parse_eval_csv(b"Query,Answer\nWhat is RAG?,Retrieval augmented generation\n")
    except ValueError as exc:
        assert str(exc) == (
            "Invalid CSV schema. Missing required columns: Expected Response. "
            "Expected columns: Query, Expected Response."
        )
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ValueError for invalid eval CSV schema")


def test_queue_chat_question_copies_chat_input_value(monkeypatch):
    fake_state = {"chat_input_value": "Hello there"}
    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)

    app.queue_chat_question()

    assert fake_state["pending_question"] == "Hello there"
