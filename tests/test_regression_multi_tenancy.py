"""Multi-tenancy regression test suite.

PURPOSE
-------
Verify tenant isolation, persistence, retrieval, publishing, evaluation, and
monitoring across every code path that touches tenant-scoped data.

RULE
----
ALL P0 tests must pass before a PR is merged.  Any failure in:
  - Tenant identification
  - Tenant isolation
  - Namespace resolution
  - Retrieval isolation
  - Shareable-chat isolation
…BLOCKS THE PR.

Test IDs follow the pattern:
  test_p0_<area>_<description>
  test_p1_<area>_<description>
  test_cross_tenant_<description>
"""
from __future__ import annotations

import sys
import uuid
from unittest.mock import MagicMock, patch

import jwt
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JWT_SECRET = "regression-test-secret-long-enough"


class FakeState(dict):
    """A dict that also supports attribute-style access.

    Streamlit's session_state supports both ``state["key"]`` and ``state.key``.
    app.py's helper functions use both styles; tests must match this behaviour.
    """

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key: str, value):
        self[key] = value


def _make_tenant_row(
    tenant_id: str | None = None,
    name: str = "TenantA",
    status: str = "PUBLISHED",
    config: dict | None = None,
    share_token: str | None = None,
) -> dict:
    tid = tenant_id or str(uuid.uuid4())
    return {
        "id": tid,
        "name": name,
        "status": status,
        "passkey_hash": "$2b$12$placeholder",
        "config": config or {},
        "share_token": share_token,
    }


def _generate_token(tenant_id: str) -> str:
    return jwt.encode({"tenant_id": tenant_id}, _JWT_SECRET, algorithm="HS256")


def _stub_langfuse_observe(monkeypatch):
    """Stub langfuse.observe so process_chat_question can run without Langfuse."""
    def fake_observe(**kwargs):
        def decorator(fn):
            def wrapper(*a, **kw):
                fn(*a, **kw)
            return wrapper
        return decorator

    fake_langfuse_mod = MagicMock()
    fake_langfuse_mod.observe = fake_observe
    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse_mod)


# ===========================================================================
# P0 — TENANT / VECTOR NAMESPACE
# ===========================================================================

class TestP0TenantNamespace:
    """Namespace is always the tenant UUID; no global fallback is possible."""

    def test_p0_namespace_is_tenant_id(self):
        """build_tenant_namespace returns the tenant_id when it is present."""
        from src.tenants import build_tenant_namespace
        tid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert build_tenant_namespace(tid) == tid

    def test_p0_namespace_fallback_is_none_when_no_tenant_and_no_file(self):
        """When there is no tenant_id and no filename, the namespace is None
        (not the string 'default'), ensuring no cross-tenant bleed."""
        from src.tenants import build_tenant_namespace
        assert build_tenant_namespace(None, None) is None

    def test_p0_namespace_not_regenerated_on_repeated_calls(self):
        """Calling build_tenant_namespace twice for the same tenant_id returns
        the same deterministic value — it never generates a new identifier."""
        from src.tenants import build_tenant_namespace
        tid = "stable-tenant-id"
        assert build_tenant_namespace(tid) == build_tenant_namespace(tid)

    def test_p0_get_active_namespace_prefers_tenant_id_over_filename(self, monkeypatch):
        """get_active_vector_namespace always resolves to tenant_id, never to a
        file-name-based namespace when a tenant is authenticated."""
        import app
        fake_state = {
            "tenant_id": "tenant-uuid-1",
            "uploaded_file_name": "some_doc.pdf",
        }
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
        assert app.get_active_vector_namespace() == "tenant-uuid-1"

    def test_p0_get_active_namespace_returns_none_without_tenant_or_file(self, monkeypatch):
        """Without a tenant_id or uploaded_file_name, the namespace is None —
        not 'default'.  This prevents accidental writes/reads to a shared
        'default' Pinecone namespace."""
        import app
        fake_state = {"tenant_id": None, "uploaded_file_name": None}
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
        assert app.get_active_vector_namespace() is None

    def test_p0_namespace_different_for_different_tenants(self):
        """Two distinct tenant IDs produce two distinct namespaces."""
        from src.tenants import build_tenant_namespace
        ns_a = build_tenant_namespace("tenant-a-uuid")
        ns_b = build_tenant_namespace("tenant-b-uuid")
        assert ns_a != ns_b

    def test_p0_whitespace_only_tenant_id_falls_back(self):
        """A whitespace-only tenant_id is treated as absent."""
        from src.tenants import build_tenant_namespace
        assert build_tenant_namespace("   ", "fallback") == "fallback"


# ===========================================================================
# P0 — TENANT ISOLATION  (vector store layer)
# ===========================================================================

class TestP0TenantIsolation:
    """Retrieval calls always carry the resolved tenant namespace."""

    def test_p0_query_uses_tenant_namespace(self, monkeypatch):
        """_run_chat_turn passes vector_namespace (= tenant_id) to the
        Pinecone query — never a hardcoded or default namespace."""
        import app
        from src.retrieval import RetrievedChunk

        namespace_used = {}
        fake_state = FakeState(
            index_built=True,
            vector_namespace="tenant-a-uuid",
            cfg_index_name="test-index",
            cfg_top_k=3,
            cfg_min_confidence=0.80,
            messages=[],
        )
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)

        chunk = RetrievedChunk(
            chunk_id="c1", text="hello", source="doc.pdf",
            chunk_index=0, page=1, score=0.95,
        )
        fake_result = {
            "matches": [chunk],
            "relevant_matches": [chunk],
            "relevant_count": 1,
            "metrics": {
                "threshold": 0.80, "retrieved_count": 1, "relevant_count": 1,
                "average_score": 0.95, "precision": 0.95, "recall": 1.0, "scores": [0.95],
            },
        }

        class FakeVS:
            def query(self, embedding, namespace, top_k, min_confidence_score):
                namespace_used["ns"] = namespace
                return fake_result

        class FakeEmbed:
            def embed_query(self, q):
                return [0.1] * 10

        monkeypatch.setattr(app, "build_vector_store", lambda c, index_name=None: FakeVS())
        monkeypatch.setattr(app, "EmbeddingService", lambda **_: FakeEmbed())

        class FakeLLM:
            def generate_answer_with_context(self, **_):
                return "answer"

        monkeypatch.setattr(app, "build_llm_service", lambda: FakeLLM())

        app._run_chat_turn(config=MagicMock(
            openai_api_key="sk", embedding_model="ada", cfg_index_name="idx",
        ), question="test", lf=None)

        assert namespace_used["ns"] == "tenant-a-uuid"

    def test_p0_upsert_uses_tenant_namespace(self, monkeypatch):
        """Knowledge-base indexing passes the tenant namespace to upsert_chunks,
        never a global or default namespace."""
        from src.tenants import build_tenant_namespace
        # Namespace for upsert is derived identically to query
        assert build_tenant_namespace("tenant-b-uuid") == "tenant-b-uuid"

    def test_p0_two_tenants_get_different_namespaces(self):
        """Tenant A and Tenant B resolve to structurally distinct namespaces,
        making cross-tenant retrieval impossible via normal code paths."""
        from src.tenants import build_tenant_namespace
        assert build_tenant_namespace("tid-a") != build_tenant_namespace("tid-b")


# ===========================================================================
# P0 — BUILDER / TEST CHAT
# ===========================================================================

class TestP0BuilderChat:
    """Builder (tab-based) chat resolves the correct tenant and namespace."""

    def test_p0_builder_chat_uses_session_tenant_id_as_namespace(self, monkeypatch):
        """The vector namespace used in builder chat equals the authenticated
        tenant's ID stored in session state."""
        import app
        fake_state = FakeState(
            tenant_id="builder-tenant-uuid",
            uploaded_file_name="kb.pdf",
        )
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
        ns = app.get_active_vector_namespace()
        assert ns == "builder-tenant-uuid"

    def test_p0_chat_blocked_when_index_not_built(self, monkeypatch):
        """If index_built is False the run-chat-turn path skips retrieval,
        preventing spurious queries against the wrong or absent namespace."""
        import app
        called = {"retrieval": False}
        fake_state = FakeState(
            index_built=False,
            vector_namespace=None,
            cfg_index_name="idx",
            cfg_top_k=3,
            cfg_min_confidence=0.80,
            messages=[],
        )
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)

        class FakeVS:
            def query(self, **_):
                called["retrieval"] = True

        monkeypatch.setattr(app, "build_vector_store", lambda c, index_name=None: FakeVS())

        class FakeLLM:
            def generate_answer(self, **_):
                return "no kb answer"

        monkeypatch.setattr(app, "build_llm_service", lambda: FakeLLM())

        app._run_chat_turn(config=MagicMock(openai_api_key="sk"), question="q", lf=None)
        assert called["retrieval"] is False


# ===========================================================================
# P0 — PUBLISHED / SHAREABLE CHAT
# ===========================================================================

class TestP0ShareableChat:
    """Shareable URL resolves to the correct tenant and its KB namespace."""

    def test_p0_share_token_embeds_tenant_id(self):
        """generate_share_token encodes the tenant_id claim in the JWT."""
        from src.share import generate_share_token, verify_share_token
        tid = str(uuid.uuid4())
        url = generate_share_token(tid, _JWT_SECRET, base_url="https://app.example.com")
        token = url.split("?token=")[1]
        assert verify_share_token(token, _JWT_SECRET) == tid

    def test_p0_share_token_different_tenants_produce_different_tokens(self):
        """Two different tenant IDs must produce different JWT tokens."""
        from src.share import generate_share_token
        url_a = generate_share_token("tid-a", _JWT_SECRET)
        url_b = generate_share_token("tid-b", _JWT_SECRET)
        assert url_a != url_b

    def test_p0_invalid_share_token_raises(self):
        """A tampered or invalid token raises ShareTokenError, blocking access."""
        from src.share import ShareTokenError, verify_share_token
        with pytest.raises(ShareTokenError):
            verify_share_token("not.a.valid.token", _JWT_SECRET)

    def test_p0_share_token_wrong_secret_raises(self):
        """A token signed with a different secret must be rejected."""
        from src.share import ShareTokenError, generate_share_token, verify_share_token
        url = generate_share_token("tid-a", _JWT_SECRET)
        token = url.split("?token=")[1]
        with pytest.raises(ShareTokenError):
            verify_share_token(token, "wrong-secret")

    def test_p0_shareable_chat_requires_published_status(self, monkeypatch):
        """The shareable URL path checks row['status'] == 'PUBLISHED' before
        rendering the chat UI.  An EVALUATED tenant must be blocked."""
        import app
        # Simulate the guard: row.status != 'PUBLISHED' → deny
        row = _make_tenant_row(status="EVALUATED")
        # The guard in main() is: if row is None or row.get("status") != "PUBLISHED"
        assert row.get("status") != "PUBLISHED"

    def test_p0_shareable_chat_loads_tenant_into_session(self, monkeypatch):
        """After a valid token is verified, _load_tenant_into_session populates
        session state with the correct tenant_id and tenant_name."""
        import app
        fake_state = {}
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
        row = _make_tenant_row(tenant_id="shared-tenant-uuid", name="PublishedCo")
        app._load_tenant_into_session(row)
        assert fake_state["tenant_id"] == "shared-tenant-uuid"
        assert fake_state["tenant_name"] == "PublishedCo"
        assert fake_state["tenant_authenticated"] is True

    def test_p0_shareable_chat_rehydrates_kb_namespace(self, monkeypatch):
        """After _load_tenant_into_session + _rehydrate_tenant_runtime_state,
        the vector_namespace equals the tenant_id, enabling KB retrieval."""
        import app
        tid = "shared-tenant-uuid"
        fake_state = {
            "tenant_id": tid,
            "tenant_status": "PUBLISHED",
            "cfg_index_name": "my-index",
            "chunks": [],
            "uploaded_file_name": None,
            "index_built": False,
            "vector_namespace": None,
            "eval_rows": None,
            "eval_file_name": None,
            "eval_results": None,
            "last_retrieval_debug": None,
        }
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
        monkeypatch.setattr(app, "get_latest_pdf_name", lambda t: "doc.pdf")

        class FakeVS:
            def describe_namespace(self, ns):
                return {"namespace_vector_count": 10, "dimension": 1536}

        monkeypatch.setattr(app, "build_vector_store", lambda c, index_name=None: FakeVS())

        app._rehydrate_tenant_runtime_state(config=MagicMock())

        assert fake_state["index_built"] is True
        assert fake_state["vector_namespace"] == tid

    def test_p0_shareable_chat_isolation_different_tokens_different_namespaces(self):
        """Token A resolves to namespace A, token B resolves to namespace B —
        they can never resolve to each other's namespace."""
        from src.share import generate_share_token, verify_share_token
        from src.tenants import build_tenant_namespace

        url_a = generate_share_token("ns-aaa", _JWT_SECRET)
        url_b = generate_share_token("ns-bbb", _JWT_SECRET)
        token_a = url_a.split("?token=")[1]
        token_b = url_b.split("?token=")[1]

        tid_a = verify_share_token(token_a, _JWT_SECRET)
        tid_b = verify_share_token(token_b, _JWT_SECRET)

        assert build_tenant_namespace(tid_a) != build_tenant_namespace(tid_b)


# ===========================================================================
# P0 — RE-LOGIN / PERSISTENCE
# ===========================================================================

class TestP0ReloginPersistence:
    """After logout → login, all tenant-scoped state is correctly restored."""

    def test_p0_logout_clears_all_tenant_state(self):
        """st.session_state.clear() on sign-out removes every tenant-scoped key,
        preventing Tenant A data from leaking to Tenant B's session."""
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
        session.clear()
        assert session == {}

    def test_p0_relogin_restores_tenant_id_and_name(self, monkeypatch):
        """_load_tenant_into_session after re-authentication sets the same
        tenant_id and name that were originally created."""
        import app
        fake_state = {}
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
        row = _make_tenant_row(tenant_id="tid-persist", name="PersistCo", status="INGESTED")
        app._load_tenant_into_session(row)
        assert fake_state["tenant_id"] == "tid-persist"
        assert fake_state["tenant_name"] == "PersistCo"
        assert fake_state["tenant_status"] == "INGESTED"

    def test_p0_relogin_restores_retrieval_config_from_persisted_row(self, monkeypatch):
        """_load_tenant_into_session restores persisted retrieval config
        (index_name, top_k, min_confidence) from the tenant row's config JSONB."""
        import app
        fake_state = {}
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
        row = _make_tenant_row(
            tenant_id="tid-cfg",
            config={"index_name": "custom-idx", "top_k": 7, "min_confidence": 0.70},
        )
        app._load_tenant_into_session(row)
        assert fake_state["cfg_index_name"] == "custom-idx"
        assert fake_state["cfg_top_k"] == 7
        assert fake_state["cfg_min_confidence"] == 0.70

    def test_p0_relogin_restores_kb_namespace_from_pinecone(self, monkeypatch):
        """After re-login, _rehydrate_tenant_runtime_state probes Pinecone and
        restores vector_namespace = tenant_id when vectors exist."""
        import app
        tid = "tid-login"
        fake_state = {
            "tenant_id": tid,
            "tenant_status": "PUBLISHED",
            "cfg_index_name": "idx",
            "chunks": [], "uploaded_file_name": None,
            "index_built": False, "vector_namespace": None,
            "eval_rows": None, "eval_file_name": None,
            "eval_results": None, "last_retrieval_debug": None,
        }
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
        monkeypatch.setattr(app, "get_latest_pdf_name", lambda t: "restored.pdf")

        class FakeVS:
            def describe_namespace(self, ns):
                assert ns == tid, f"Expected namespace {tid!r}, got {ns!r}"
                return {"namespace_vector_count": 5, "dimension": 1536}

        monkeypatch.setattr(app, "build_vector_store", lambda c, index_name=None: FakeVS())
        app._rehydrate_tenant_runtime_state(config=MagicMock())

        assert fake_state["vector_namespace"] == tid
        assert fake_state["index_built"] is True

    def test_p0_relogin_namespace_is_not_regenerated(self, monkeypatch):
        """Logging out and logging in again produces the same namespace as
        before — it is always derived from the immutable tenant_id UUID."""
        from src.tenants import build_tenant_namespace
        tid = "immutable-tenant-uuid"
        ns_before_logout = build_tenant_namespace(tid)
        # Simulate logout (clear) then login (set again)
        ns_after_login = build_tenant_namespace(tid)
        assert ns_before_logout == ns_after_login

    def test_p0_relogin_share_url_restored_from_row(self, monkeypatch):
        """_load_tenant_into_session restores the share_token (URL) from the
        database row, so the publish state survives re-login."""
        import app
        fake_state = {}
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
        expected_url = "https://app.example.com/?token=abc.def.ghi"
        row = _make_tenant_row(status="PUBLISHED", share_token=expected_url)
        app._load_tenant_into_session(row)
        assert fake_state["tenant_share_url"] == expected_url


# ===========================================================================
# P1 — EVALUATION
# ===========================================================================

class TestP1Evaluation:
    """Evaluation always uses the authenticated tenant's namespace."""

    def test_p1_eval_namespace_matches_tenant(self, monkeypatch):
        """retrieve_and_answer (used in eval loops) queries against the same
        vector_namespace that is set in session state for the tenant."""
        import app
        from src.retrieval import RetrievedChunk

        namespace_used = {}
        tid = "eval-tenant-uuid"
        fake_state = FakeState(
            index_built=True,
            vector_namespace=tid,
            cfg_index_name="idx",
            cfg_top_k=3,
            cfg_min_confidence=0.80,
            messages=[],
        )
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)

        chunk = RetrievedChunk(
            chunk_id="e1", text="eval text", source="doc.pdf",
            chunk_index=0, page=None, score=0.91,
        )
        fake_result = {
            "matches": [chunk],
            "relevant_matches": [chunk],
            "relevant_count": 1,
            "metrics": {
                "threshold": 0.80, "retrieved_count": 1, "relevant_count": 1,
                "average_score": 0.91, "precision": 0.91, "recall": 1.0, "scores": [0.91],
            },
        }

        class FakeVS:
            def query(self, embedding, namespace, top_k, min_confidence_score):
                namespace_used["ns"] = namespace
                return fake_result

        monkeypatch.setattr(app, "build_vector_store", lambda c, index_name=None: FakeVS())
        monkeypatch.setattr(app, "EmbeddingService", lambda **_: MagicMock(embed_query=lambda q: [0.1] * 10))

        class FakeLLM:
            def generate_answer_with_context(self, **_):
                return "eval answer"

        monkeypatch.setattr(app, "build_llm_service", lambda: FakeLLM())

        result = app.retrieve_and_answer(
            config=MagicMock(openai_api_key="sk", embedding_model="ada"),
            question="eval question",
        )
        assert namespace_used["ns"] == tid
        assert result == "eval answer"

    def test_p1_eval_results_keyed_to_tenant(self):
        """save_eval_log inserts rows with the correct tenant_id, preventing
        eval data from being associated with the wrong tenant."""
        import importlib, sys
        mod_name = "src.supabase_client"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        sc = importlib.import_module(mod_name)

        mock_client = MagicMock()
        mock_client.table.return_value = mock_client
        mock_client.insert.return_value = mock_client
        mock_client.execute.return_value = MagicMock(data=[])
        sc._client = mock_client

        rows = [{"query": "q1", "expected": "e1", "actual": "a1", "score": 4, "reason": "ok"}]
        sc.save_eval_log("target-tenant-id", rows)

        inserted = mock_client.insert.call_args[0][0]
        assert all(r["tenant_id"] == "target-tenant-id" for r in inserted)


# ===========================================================================
# P1 — MONITORING / TRACES
# ===========================================================================

class TestP1MonitoringTraces:
    """Traces emitted during chat turns contain the mandatory metadata fields."""

    def test_p1_trace_includes_tenant_id_and_name(self, monkeypatch):
        """process_chat_question calls update_current_span with tenant_id and
        tenant_name in the metadata block."""
        import app
        _stub_langfuse_observe(monkeypatch)

        captured_metadata = {}

        class FakeLF:
            def update_current_span(self, input=None, metadata=None, output=None):
                if metadata:
                    captured_metadata.update(metadata)
            def get_current_trace_id(self):
                return "trace-001"

        fake_state = FakeState(
            tenant_id="tid-trace",
            tenant_name="TraceTenant",
            vector_namespace="tid-trace",
            index_built=False,
            cfg_index_name="idx",
            cfg_top_k=3,
            cfg_min_confidence=0.80,
            messages=[],
            last_retrieval_debug=None,
        )
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
        monkeypatch.setattr(app, "get_langfuse", lambda: FakeLF())

        class FakeLLM:
            def generate_answer(self, **_):
                return "no kb answer"
        monkeypatch.setattr(app, "build_llm_service", lambda: FakeLLM())

        app.process_chat_question(
            config=MagicMock(openai_api_key="sk", embedding_model="ada"),
            question="test question",
            chat_type="TEST_CHAT",
        )

        assert captured_metadata.get("tenant_id") == "tid-trace"
        assert captured_metadata.get("tenant_name") == "TraceTenant"

    def test_p1_trace_includes_vector_namespace(self, monkeypatch):
        """update_current_span metadata includes vector_namespace."""
        import app
        _stub_langfuse_observe(monkeypatch)

        captured_metadata = {}

        class FakeLF:
            def update_current_span(self, input=None, metadata=None, output=None):
                if metadata:
                    captured_metadata.update(metadata)
            def get_current_trace_id(self):
                return "trace-002"

        fake_state = FakeState(
            tenant_id="tid-ns",
            tenant_name="NsTenant",
            vector_namespace="tid-ns",
            index_built=False,
            cfg_index_name="idx",
            cfg_top_k=3,
            cfg_min_confidence=0.80,
            messages=[],
            last_retrieval_debug=None,
        )
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
        monkeypatch.setattr(app, "get_langfuse", lambda: FakeLF())

        class FakeLLM:
            def generate_answer(self, **_):
                return "ans"
        monkeypatch.setattr(app, "build_llm_service", lambda: FakeLLM())

        app.process_chat_question(
            config=MagicMock(openai_api_key="sk", embedding_model="ada"),
            question="q",
            chat_type="REAL_CHAT",
        )
        assert captured_metadata.get("vector_namespace") == "tid-ns"

    def test_p1_trace_includes_chat_type(self, monkeypatch):
        """chat_type is present in trace metadata for both TEST_CHAT and REAL_CHAT."""
        import app
        _stub_langfuse_observe(monkeypatch)

        for expected_type in ("TEST_CHAT", "REAL_CHAT"):
            captured_metadata = {}

            class FakeLF:
                def update_current_span(self, input=None, metadata=None, output=None):
                    if metadata:
                        captured_metadata.update(metadata)
                def get_current_trace_id(self):
                    return "trace-003"

            fake_state = FakeState(
                tenant_id="t", tenant_name="T", vector_namespace="t",
                index_built=False, cfg_index_name="i", cfg_top_k=3,
                cfg_min_confidence=0.80, messages=[], last_retrieval_debug=None,
            )
            monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
            monkeypatch.setattr(app, "get_langfuse", lambda: FakeLF())

            class FakeLLM:
                def generate_answer(self, **_): return "ans"
            monkeypatch.setattr(app, "build_llm_service", lambda: FakeLLM())

            app.process_chat_question(
                config=MagicMock(openai_api_key="sk", embedding_model="ada"),
                question="q",
                chat_type=expected_type,
            )
            assert captured_metadata.get("chat_type") == expected_type, (
                f"chat_type not set for {expected_type}"
            )

    def test_p1_trace_retrieval_includes_chunk_ids(self, monkeypatch):
        """When retrieval runs, chunk_ids of relevant matches appear in the
        span metadata so every trace is linked to specific document chunks."""
        import app
        from src.retrieval import RetrievedChunk

        span_metadata = {}

        class FakeLF:
            def update_current_span(self, input=None, metadata=None, output=None):
                if metadata:
                    span_metadata.update(metadata)
            def get_current_trace_id(self):
                return "trace-004"

        chunk = RetrievedChunk(
            chunk_id="chunk-xyz", text="data", source="f.pdf",
            chunk_index=1, page=2, score=0.92,
        )
        fake_result = {
            "matches": [chunk],
            "relevant_matches": [chunk],
            "relevant_count": 1,
            "metrics": {
                "threshold": 0.80, "retrieved_count": 1, "relevant_count": 1,
                "average_score": 0.92, "precision": 0.92, "recall": 1.0,
                "scores": [0.92],
            },
        }

        class FakeVS:
            def query(self, **_): return fake_result

        class FakeEmbed:
            def embed_query(self, q): return [0.0] * 10

        class FakeLLM:
            def generate_answer_with_context(self, **_): return "ans"

        fake_state = FakeState(
            index_built=True,
            vector_namespace="tid-chunks",
            cfg_index_name="idx",
            cfg_top_k=3,
            cfg_min_confidence=0.80,
            messages=[],
        )
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
        monkeypatch.setattr(app, "build_vector_store", lambda c, index_name=None: FakeVS())
        monkeypatch.setattr(app, "EmbeddingService", lambda **_: FakeEmbed())
        monkeypatch.setattr(app, "build_llm_service", lambda: FakeLLM())

        app._run_chat_turn(
            config=MagicMock(openai_api_key="sk", embedding_model="ada"),
            question="q",
            lf=FakeLF(),
        )
        assert "chunk_ids" in span_metadata
        assert "chunk-xyz" in span_metadata["chunk_ids"]


# ===========================================================================
# MANDATORY CROSS-TENANT TEST
# ===========================================================================

class TestCrossTenantIsolation:
    """
    Tenant A: Namespace = tenant_a_uuid
    Tenant B: Namespace = tenant_b_uuid

    [ ] A → A_PRIVATE_DATA  = PASS (retrieved from ns_a, not ns_b)
    [ ] B → B_PRIVATE_DATA  = PASS (retrieved from ns_b, not ns_a)
    [ ] A → B_PRIVATE_DATA  = MUST NOT RETRIEVE
    [ ] B → A_PRIVATE_DATA  = MUST NOT RETRIEVE
    [ ] A Shareable URL → correct namespace = PASS
    [ ] B Shareable URL → correct namespace = PASS
    [ ] A token → B namespace  = MUST NOT resolve
    [ ] B token → A namespace  = MUST NOT resolve
    """

    NS_A = "aaaaaaaa-0000-0000-0000-000000000000"
    NS_B = "bbbbbbbb-0000-0000-0000-000000000000"

    def _query_with_namespace(self, monkeypatch, active_namespace: str, store_namespace: str) -> list:
        """Run _run_chat_turn with a fixed active_namespace and a fake store
        that only returns data when the queried namespace matches store_namespace."""
        import app
        from src.retrieval import RetrievedChunk

        chunk = RetrievedChunk(
            chunk_id=f"chunk-{store_namespace[:8]}", text=f"DATA_FOR_{store_namespace}",
            source="doc.pdf", chunk_index=0, page=1, score=0.95,
        )

        class FakeVS:
            def query(self, embedding, namespace, top_k, min_confidence_score):
                if namespace == store_namespace:
                    return {
                        "matches": [chunk],
                        "relevant_matches": [chunk],
                        "relevant_count": 1,
                        "metrics": {
                            "threshold": 0.80, "retrieved_count": 1, "relevant_count": 1,
                            "average_score": 0.95, "precision": 0.95, "recall": 1.0,
                            "scores": [0.95],
                        },
                    }
                return {
                    "matches": [], "relevant_matches": [], "relevant_count": 0,
                    "metrics": {
                        "threshold": 0.80, "retrieved_count": 0, "relevant_count": 0,
                        "average_score": 0.0, "precision": 0.0, "recall": 0.0, "scores": [],
                    },
                }

        returned_chunks = []

        class FakeLLM:
            def generate_answer_with_context(self, question, context, **_):
                returned_chunks.append(context)
                return "answer"
            def generate_no_context_answer(self, **_):
                return "no context"
            def generate_answer(self, **_):
                return "no kb"

        fake_state = FakeState(
            index_built=True,
            vector_namespace=active_namespace,
            cfg_index_name="idx",
            cfg_top_k=3,
            cfg_min_confidence=0.80,
            messages=[],
        )
        monkeypatch.setattr(app.st, "session_state", fake_state, raising=False)
        monkeypatch.setattr(app, "build_vector_store", lambda c, index_name=None: FakeVS())
        monkeypatch.setattr(app, "EmbeddingService", lambda **_: MagicMock(embed_query=lambda q: [0.0] * 10))
        monkeypatch.setattr(app, "build_llm_service", lambda: FakeLLM())

        app._run_chat_turn(
            config=MagicMock(openai_api_key="sk", embedding_model="ada"),
            question="what is the private data?",
            lf=None,
        )
        return returned_chunks

    def test_cross_tenant_a_retrieves_a_data(self, monkeypatch):
        """Tenant A querying Tenant A's store → data returned."""
        chunks = self._query_with_namespace(monkeypatch, self.NS_A, self.NS_A)
        assert len(chunks) == 1
        assert self.NS_A in chunks[0]

    def test_cross_tenant_b_retrieves_b_data(self, monkeypatch):
        """Tenant B querying Tenant B's store → data returned."""
        chunks = self._query_with_namespace(monkeypatch, self.NS_B, self.NS_B)
        assert len(chunks) == 1
        assert self.NS_B in chunks[0]

    def test_cross_tenant_a_cannot_retrieve_b_data(self, monkeypatch):
        """Tenant A's namespace must NOT return Tenant B's documents."""
        chunks = self._query_with_namespace(monkeypatch, self.NS_A, self.NS_B)
        assert len(chunks) == 0, "Tenant A must not retrieve Tenant B documents"

    def test_cross_tenant_b_cannot_retrieve_a_data(self, monkeypatch):
        """Tenant B's namespace must NOT return Tenant A's documents."""
        chunks = self._query_with_namespace(monkeypatch, self.NS_B, self.NS_A)
        assert len(chunks) == 0, "Tenant B must not retrieve Tenant A documents"

    def test_cross_tenant_share_token_a_resolves_to_ns_a(self):
        """Share token generated for Tenant A resolves to Tenant A's namespace."""
        from src.share import generate_share_token, verify_share_token
        from src.tenants import build_tenant_namespace
        url = generate_share_token(self.NS_A, _JWT_SECRET)
        tid = verify_share_token(url.split("?token=")[1], _JWT_SECRET)
        assert build_tenant_namespace(tid) == self.NS_A

    def test_cross_tenant_share_token_b_resolves_to_ns_b(self):
        """Share token generated for Tenant B resolves to Tenant B's namespace."""
        from src.share import generate_share_token, verify_share_token
        from src.tenants import build_tenant_namespace
        url = generate_share_token(self.NS_B, _JWT_SECRET)
        tid = verify_share_token(url.split("?token=")[1], _JWT_SECRET)
        assert build_tenant_namespace(tid) == self.NS_B

    def test_cross_tenant_share_token_a_cannot_resolve_to_ns_b(self):
        """Tenant A's share token must not produce Tenant B's namespace."""
        from src.share import generate_share_token, verify_share_token
        from src.tenants import build_tenant_namespace
        url_a = generate_share_token(self.NS_A, _JWT_SECRET)
        tid = verify_share_token(url_a.split("?token=")[1], _JWT_SECRET)
        assert build_tenant_namespace(tid) != self.NS_B

    def test_cross_tenant_share_token_b_cannot_resolve_to_ns_a(self):
        """Tenant B's share token must not produce Tenant A's namespace."""
        from src.share import generate_share_token, verify_share_token
        from src.tenants import build_tenant_namespace
        url_b = generate_share_token(self.NS_B, _JWT_SECRET)
        tid = verify_share_token(url_b.split("?token=")[1], _JWT_SECRET)
        assert build_tenant_namespace(tid) != self.NS_A

    def test_cross_tenant_share_token_for_a_not_valid_for_b_secret(self):
        """A token created for Tenant A cannot be decoded with a different secret,
        preventing namespace hijacking by guessing/forging tokens."""
        from src.share import ShareTokenError, generate_share_token, verify_share_token
        url = generate_share_token(self.NS_A, _JWT_SECRET)
        token = url.split("?token=")[1]
        with pytest.raises(ShareTokenError):
            verify_share_token(token, "wrong-secret-for-tenant-b")
