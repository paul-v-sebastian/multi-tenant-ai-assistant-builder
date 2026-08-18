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
    assert fake_state["tenant_id"] is None
    assert fake_state["tenant_status"] is None
    assert fake_state["tenant_authenticated"] is False
    assert fake_state["tenant_share_url"] is None


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


def test_get_active_vector_namespace_prefers_tenant_id(monkeypatch):
    fake_state = {"tenant_id": "tenant-123", "uploaded_file_name": "sample.pdf"}
    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)

    assert app.get_active_vector_namespace() == "tenant-123"


def test_get_active_vector_namespace_falls_back_to_uploaded_file(monkeypatch):
    fake_state = {"tenant_id": None, "uploaded_file_name": "sample.pdf"}
    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)

    assert app.get_active_vector_namespace() == "sample.pdf"


def test_rehydrate_tenant_runtime_state_restores_persisted_kb(monkeypatch):
    fake_state = {
        "tenant_id": "tenant-123",
        "cfg_index_name": "custom-index",
        "chunks": ["stale"],
        "uploaded_file_name": "old.pdf",
        "index_built": False,
        "vector_namespace": None,
        "eval_rows": [{"Query": "q", "Expected Response": "e"}],
        "eval_file_name": "eval.csv",
        "eval_results": [{"Score (1-5)": 5}],
        "last_retrieval_debug": {"Retrieved": "1"},
    }
    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
    monkeypatch.setattr(app, "get_latest_pdf_name", lambda tenant_id: "persisted.pdf")

    class FakeVectorStore:
        def describe_namespace(self, namespace):
            assert namespace == "tenant-123"
            return {"namespace_vector_count": 4, "dimension": 1536}

    captured = {}

    def fake_build_vector_store(config, index_name=None):
        captured["index_name"] = index_name
        return FakeVectorStore()

    monkeypatch.setattr(app, "build_vector_store", fake_build_vector_store)

    app._rehydrate_tenant_runtime_state(config=object())

    assert captured["index_name"] == "custom-index"
    assert fake_state["chunks"] == []
    assert fake_state["uploaded_file_name"] == "persisted.pdf"
    assert fake_state["index_built"] is True
    assert fake_state["vector_namespace"] == "tenant-123"
    assert fake_state["eval_rows"] is None
    assert fake_state["eval_file_name"] is None
    assert fake_state["eval_results"] is None
    assert fake_state["last_retrieval_debug"] is None


def test_rehydrate_tenant_runtime_state_leaves_empty_kb_when_no_persisted_data(monkeypatch):
    fake_state = {
        "tenant_id": "tenant-123",
        "cfg_index_name": "custom-index",
        "chunks": ["stale"],
        "uploaded_file_name": "old.pdf",
        "index_built": True,
        "vector_namespace": "old-ns",
        "eval_rows": [{"Query": "q", "Expected Response": "e"}],
        "eval_file_name": "eval.csv",
        "eval_results": [{"Score (1-5)": 5}],
        "last_retrieval_debug": {"Retrieved": "1"},
    }
    monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
    monkeypatch.setattr(app, "get_latest_pdf_name", lambda tenant_id: None)

    class FakeVectorStore:
        def describe_namespace(self, namespace):
            return {"namespace_vector_count": 0, "dimension": 1536}

    monkeypatch.setattr(app, "build_vector_store", lambda config, index_name=None: FakeVectorStore())

    app._rehydrate_tenant_runtime_state(config=object())

    assert fake_state["chunks"] == []
    assert fake_state["uploaded_file_name"] is None
    assert fake_state["index_built"] is False
    assert fake_state["vector_namespace"] is None
    assert fake_state["eval_rows"] is None
    assert fake_state["eval_file_name"] is None
    assert fake_state["eval_results"] is None
    assert fake_state["last_retrieval_debug"] is None


# ---------------------------------------------------------------------------
# Bug 2: session state is fully cleared on logout
# ---------------------------------------------------------------------------

def test_logout_clears_all_session_state():
    """st.session_state.clear() removes every key including tenant-scoped data."""
    # Simulate a fully-populated session from Tenant A
    session = {
        "tenant_authenticated": True,
        "tenant_id": "tid-A",
        "tenant_name": "TenantA",
        "tenant_status": "PUBLISHED",
        "tenant_share_url": "https://app/?token=xyz",
        "messages": [{"role": "user", "content": "hello"}],
        "chunks": ["chunk1"],
        "uploaded_file_name": "doc.pdf",
        "index_built": True,
        "vector_namespace": "tid-A",
        "eval_rows": [{"Query": "q", "Expected Response": "e"}],
        "eval_results": [{"Score (1-5)": 5}],
        "cfg_index_name": "custom-index",
    }

    # This is what the sign-out handler now does
    session.clear()

    assert session == {}
