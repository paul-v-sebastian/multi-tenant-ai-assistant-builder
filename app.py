from __future__ import annotations

import csv
import io
import json

import streamlit as st
from openai import OpenAI

from src.config import load_config
from src.embeddings import EmbeddingService, EmbeddingServiceError
from src.llm import LLMService, LLMServiceError
from src.pdf_processor import PDFProcessingError, build_chunks, extract_pdf_pages
from src.tenants import build_tenant_namespace
from src.retrieval import format_citation
from src.tracing import get_langfuse, init_langfuse
from src.vector_store import PineconeVectorStore, VectorStoreError
from src.supabase_client import (
    get_supabase_status,
    init_supabase,
    create_tenant,
    get_tenant,
    get_tenant_by_name,
    verify_tenant_passkey,
    update_tenant_status,
    upsert_tenant_config,
    save_eval_log,
    SupabaseError,
)
from src.storage import upload_pdf, upload_eval_csv

_JUDGE_MODEL = "gpt-4o-mini"
_JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge. Given a query, an expected response, and an actual response, "
    "score the actual response on semantic accuracy and factual correctness on a scale of 1 to 5 "
    "(1 = completely wrong, 5 = perfect match). "
    'Return ONLY valid JSON in the format: {"score": <int 1-5>, "reason": "<brief explanation>"}'
)

_GLOBAL_CSS = """
<style>
.block-container { max-width: 900px; padding-top: 2rem; }
</style>
"""

_EVAL_REQUIRED_COLUMNS = ("Query", "Expected Response")


def initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("uploaded_file_name", None)
    st.session_state.setdefault("index_built", False)
    st.session_state.setdefault("vector_namespace", None)
    st.session_state.setdefault("tenant_id", None)
    st.session_state.setdefault("tenant_name", None)
    st.session_state.setdefault("tenant_status", None)
    st.session_state.setdefault("tenant_authenticated", False)
    st.session_state.setdefault("tenant_share_url", None)
    # Phase 5 sidebar config defaults (POC values)
    st.session_state.setdefault("cfg_index_name", "my-pdf-index")
    st.session_state.setdefault("cfg_top_k", 3)
    st.session_state.setdefault("cfg_min_confidence", 0.80)
    st.session_state.setdefault("eval_rows", None)
    st.session_state.setdefault("eval_file_name", None)
    st.session_state.setdefault("eval_results", None)
    st.session_state.setdefault("last_retrieval_debug", None)
    st.session_state.setdefault("pending_question", None)
    # Phase 5: per-message feedback state {message_index: True/False (submitted)}
    st.session_state.setdefault("feedback_given", {})


def clear_conversation() -> None:
    state = st.session_state
    if isinstance(state, dict):
        state["messages"] = []
        state["last_retrieval_debug"] = None
    else:
        state.messages = []
        state.last_retrieval_debug = None


def show_chat_error(exc: Exception) -> None:
    error_message = f"⚠️ An error occurred: {exc}"
    st.error(error_message)
    state = st.session_state
    messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
    if messages is None:
        if isinstance(state, dict):
            state["messages"] = []
            messages = state["messages"]
        else:
            state.messages = []
            messages = state.messages
    if messages and messages[-1].get("role") == "user":
        messages.pop()
    messages.append({"role": "assistant", "content": error_message})
    if isinstance(state, dict):
        state["last_retrieval_debug"] = None
    else:
        state.last_retrieval_debug = None


def build_llm_service() -> LLMService:
    config = load_config()
    return LLMService(api_key=config.openai_api_key, model=config.llm_model)


def build_vector_store(config, index_name: str | None = None) -> PineconeVectorStore:
    return PineconeVectorStore(
        api_key=config.pinecone_api_key,
        index_name=index_name or config.pinecone_index_name,
        dimension=config.embedding_dimension,
        cloud=config.pinecone_cloud,
        region=config.pinecone_region,
    )


def get_active_vector_namespace() -> str | None:
    return build_tenant_namespace(
        tenant_id=st.session_state.get("tenant_id"),
        fallback_namespace=st.session_state.get("uploaded_file_name") or "default",
    )


def parse_eval_csv(csv_bytes: bytes) -> list[dict[str, str]]:
    decoded = csv_bytes.decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))
    fieldnames = reader.fieldnames or []
    missing_columns = [column for column in _EVAL_REQUIRED_COLUMNS if column not in fieldnames]
    if missing_columns:
        raise ValueError(
            "Invalid CSV schema. Missing required columns: "
            + ", ".join(missing_columns)
            + ". Expected columns: Query, Expected Response."
        )
    return list(reader)


def queue_chat_question() -> None:
    question = st.session_state.get("chat_input_value", "")
    if question:
        st.session_state["pending_question"] = question


def process_chat_question(config, question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        lf = get_langfuse()
        from langfuse import observe  # noqa: PLC0415

        @observe(name="chat_turn", capture_input=False, capture_output=False)
        def _traced_chat_turn() -> None:
            if lf:
                lf.update_current_span(input={"question": question})
            _run_chat_turn(config, question, lf)

        _traced_chat_turn()
    except (LLMServiceError, EmbeddingServiceError, VectorStoreError, Exception) as exc:  # noqa: BLE001
        show_chat_error(exc)


def _run_chat_turn(config, question: str, lf) -> None:
    try:
        llm_service = build_llm_service()

        if st.session_state.index_built and st.session_state.vector_namespace:
            embedding_svc = EmbeddingService(
                api_key=config.openai_api_key,
                model=config.embedding_model,
            )
            question_embedding = embedding_svc.embed_query(question)
            vector_store = build_vector_store(config, index_name=st.session_state.cfg_index_name)
            retrieval_result = vector_store.query(
                embedding=question_embedding,
                namespace=st.session_state.vector_namespace,
                top_k=st.session_state.cfg_top_k,
                min_confidence_score=st.session_state.cfg_min_confidence,
            )
            metrics = retrieval_result["metrics"]
            relevant_matches = retrieval_result["relevant_matches"]

            # --- Phase 4: instrument retrieval metrics on current span ---
            if lf:
                lf.update_current_span(
                    metadata={
                        "threshold": metrics["threshold"],
                        "retrieved": metrics["retrieved_count"],
                        "relevant": metrics["relevant_count"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "scores": metrics["scores"],
                    }
                )
            # --- End Phase 4 retrieval metadata ---

            if not relevant_matches:
                answer = llm_service.generate_no_context_answer(
                    question=question,
                    conversation_history=st.session_state.messages[:-1],
                )
                citations = []
            else:
                context = "\n\n".join(
                    f"[Chunk {match.chunk_index}] {match.text}" for match in relevant_matches
                )
                answer = llm_service.generate_answer_with_context(
                    question=question,
                    context=context,
                    conversation_history=st.session_state.messages[:-1],
                )
                citations = [format_citation(match) for match in relevant_matches]

            debug_message = {
                "Threshold": f"{metrics['threshold']:.2f}",
                "Retrieved": str(metrics["retrieved_count"]),
                "Relevant": str(metrics["relevant_count"]),
                "Precision": f"{metrics['precision']:.2f}",
                "Recall": f"{metrics['recall']:.2f}",
                "Scores": ", ".join(f"{score:.2f}" for score in metrics["scores"]) if metrics["scores"] else "None",
            }
        else:
            answer = llm_service.generate_answer(
                question=question,
                conversation_history=st.session_state.messages[:-1],
            )
            citations = []
            debug_message = None

        # --- Phase 4: capture answer on span; Phase 5: capture trace_id ---
        if lf:
            lf.update_current_span(output={"answer": answer})
        trace_id = lf.get_current_trace_id() if lf else None
        # --- End Phase 4/5 ---

        message: dict = {"role": "assistant", "content": answer}
        if citations:
            message["citations"] = citations
        # Phase 5: attach trace_id to the message for feedback wiring
        if trace_id:
            message["trace_id"] = trace_id
        st.session_state["last_retrieval_debug"] = debug_message
        st.session_state.messages.append(message)
    except (LLMServiceError, EmbeddingServiceError, VectorStoreError, Exception) as exc:  # noqa: BLE001
        show_chat_error(exc)


def retrieve_and_answer(config, question: str) -> str:
    """Run retrieval + generation for *question* and return the answer string.

    Reuses the same embedding / vector-store / LLM services as the chat flow so
    that eval runs produce real, instrumented responses without duplicating logic.
    Does not touch st.session_state.messages.
    """
    llm_service = build_llm_service()
    if st.session_state.index_built and st.session_state.vector_namespace:
        embedding_svc = EmbeddingService(
            api_key=config.openai_api_key,
            model=config.embedding_model,
        )
        question_embedding = embedding_svc.embed_query(question)
        vector_store = build_vector_store(config, index_name=st.session_state.cfg_index_name)
        retrieval_result = vector_store.query(
            embedding=question_embedding,
            namespace=st.session_state.vector_namespace,
            top_k=st.session_state.cfg_top_k,
            min_confidence_score=st.session_state.cfg_min_confidence,
        )
        relevant_matches = retrieval_result["relevant_matches"]
        if not relevant_matches:
            return llm_service.generate_no_context_answer(question=question, conversation_history=[])
        context = "\n\n".join(
            f"[Chunk {match.chunk_index}] {match.text}" for match in relevant_matches
        )
        return llm_service.generate_answer_with_context(
            question=question, context=context, conversation_history=[]
        )
    return llm_service.generate_answer(question=question, conversation_history=[])


def evaluate_response_with_judge(
    openai_api_key: str,
    query: str,
    expected_response: str,
    actual_response: str,
) -> dict[str, int | str]:
    """Call the gpt-4o-mini judge and return {"score": int, "reason": str}.

    Matches the POC's evaluate_response_with_judge() logic exactly:
    same model, same prompt structure, same JSON score/reason format.
    """
    client = OpenAI(api_key=openai_api_key)
    user_message = (
        f"Query: {query}\n\n"
        f"Expected Response: {expected_response}\n\n"
        f"Actual Response: {actual_response}"
    )
    try:
        response = client.chat.completions.create(
            model=_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMServiceError(f"Judge API failure: {exc}") from exc

    raw = (response.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
        return {"score": int(parsed["score"]), "reason": str(parsed["reason"])}
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise LLMServiceError(f"Judge returned unexpected format: {raw!r}") from exc


def _render_feedback_buttons(message_index: int, trace_id: str | None) -> None:
    """Render compact 👍/👎 buttons under an assistant message.

    Feedback is stored as a binary Langfuse score (1 = good, 0 = bad) attached
    to the specific trace_id of that chat turn.  Once submitted the buttons are
    replaced by a brief confirmation so the layout is not disrupted.
    """
    feedback_given = st.session_state.feedback_given
    if feedback_given.get(message_index) is not None:
        icon = "👍" if feedback_given[message_index] else "👎"
        st.caption(f"{icon} Feedback recorded")
        return

    col_up, col_down, _ = st.columns([1, 1, 8])
    with col_up:
        if st.button("👍", key=f"fb_up_{message_index}", help="Good response"):
            _submit_feedback(message_index, trace_id, score=1)
            st.rerun()
    with col_down:
        if st.button("👎", key=f"fb_down_{message_index}", help="Bad response"):
            _submit_feedback(message_index, trace_id, score=0)
            st.rerun()


def _submit_feedback(message_index: int, trace_id: str | None, score: int) -> None:
    """Record feedback in session_state and, if Langfuse is configured, attach it to the trace."""
    st.session_state.feedback_given[message_index] = bool(score)
    lf = get_langfuse()
    if lf and trace_id:
        try:
            lf.create_score(
                trace_id=trace_id,
                name="user_feedback",
                value=score,
                data_type="NUMERIC",
                comment="thumbs_up" if score == 1 else "thumbs_down",
            )
        except Exception:  # noqa: BLE001
            pass  # Feedback failure must never break the chat UI


def _load_tenant_into_session(row: dict) -> None:
    """Populate session_state from a fetched tenant row after successful auth."""
    state = st.session_state
    if isinstance(state, dict):
        state["tenant_id"] = row["id"]
        state["tenant_name"] = row.get("name", "")
        state["tenant_status"] = row.get("status")
        state["tenant_authenticated"] = True
        state["tenant_share_url"] = row.get("share_token")
    else:
        state.tenant_id = row["id"]
        state.tenant_name = row.get("name", "")
        state.tenant_status = row.get("status")
        state.tenant_authenticated = True
        state.tenant_share_url = row.get("share_token")
    # Restore persisted retrieval config (fall back to current defaults)
    cfg = row.get("config") or {}
    if "index_name" in cfg:
        if isinstance(state, dict):
            state["cfg_index_name"] = cfg["index_name"]
        else:
            state.cfg_index_name = cfg["index_name"]
    if "top_k" in cfg:
        if isinstance(state, dict):
            state["cfg_top_k"] = int(cfg["top_k"])
        else:
            state.cfg_top_k = int(cfg["top_k"])
    if "min_confidence" in cfg:
        if isinstance(state, dict):
            state["cfg_min_confidence"] = float(cfg["min_confidence"])
        else:
            state.cfg_min_confidence = float(cfg["min_confidence"])


def render_tenant_tab() -> None:
    """Tenant selection / creation screen (Phase 3)."""
    st.subheader("Tenant Management")

    if st.session_state.tenant_authenticated:
        st.success(
            f"✅ Authenticated as **{st.session_state.tenant_name}** "
            f"(ID: `{st.session_state.tenant_id}`)"
        )
        if st.button("Sign out", key="tenant_signout"):
            st.session_state.clear()
            st.rerun()
        return

    supabase_status, _ = get_supabase_status()
    if supabase_status != "green":
        st.warning(
            "⚠️ Supabase is not connected. Configure `SUPABASE_URL` and "
            "`SUPABASE_KEY` to enable tenant management."
        )
        return

    flow = st.radio(
        "What would you like to do?",
        ["Select existing tenant", "Create new tenant"],
        key="tenant_flow_radio",
        horizontal=True,
    )

    if flow == "Create new tenant":
        st.markdown("#### Create a new tenant")
        new_name = st.text_input("Tenant name", key="new_tenant_name")
        new_passkey = st.text_input(
            "Passkey", type="password", key="new_tenant_passkey",
            help="Choose a strong passkey — it will be bcrypt-hashed before storage."
        )
        new_passkey_confirm = st.text_input(
            "Confirm passkey", type="password", key="new_tenant_passkey_confirm"
        )

        if st.button("Create tenant", key="btn_create_tenant"):
            if not new_name.strip():
                st.error("Tenant name cannot be empty.")
            elif not new_passkey:
                st.error("Passkey cannot be empty.")
            elif new_passkey != new_passkey_confirm:
                st.error("Passkeys do not match.")
            else:
                import bcrypt  # noqa: PLC0415
                passkey_hash = bcrypt.hashpw(
                    new_passkey.encode(), bcrypt.gensalt()
                ).decode()
                try:
                    tenant_id = create_tenant(new_name.strip(), passkey_hash)
                    row = get_tenant(tenant_id)
                    if row:
                        _load_tenant_into_session(row)
                        st.success(f"Tenant created! Your tenant ID: `{tenant_id}`")
                        st.info("📋 Save this ID — you will need it to sign in next time.")
                        st.rerun()
                except SupabaseError as exc:
                    st.error(f"Failed to create tenant: {exc}")

    else:  # Select existing tenant
        st.markdown("#### Sign in to an existing tenant")
        existing_name = st.text_input("Tenant name", key="existing_tenant_name")
        existing_passkey = st.text_input(
            "Passkey", type="password", key="existing_tenant_passkey"
        )

        if st.button("Sign in", key="btn_signin_tenant"):
            if not existing_name.strip():
                st.error("Tenant name cannot be empty.")
            elif not existing_passkey:
                st.error("Passkey cannot be empty.")
            else:
                try:
                    row = get_tenant_by_name(existing_name.strip())
                    if row is None:
                        st.error("Tenant not found. Please check the name and try again.")
                    else:
                        ok = verify_tenant_passkey(row["id"], existing_passkey)
                        if ok:
                            _load_tenant_into_session(row)
                            st.success(f"Welcome back, **{row.get('name', '')}**!")
                            st.rerun()
                        else:
                            st.error("Incorrect passkey. Please try again.")
                except SupabaseError as exc:
                    st.error(f"Authentication error. Please try again.")


def render_knowledge_base_tab(config) -> None:
    st.subheader("Knowledge Base")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="kb_pdf_uploader")
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.uploaded_file_name:
            try:
                pdf_bytes = uploaded_file.read()
                pages = extract_pdf_pages(pdf_bytes)
                chunks = build_chunks(
                    pages,
                    source=uploaded_file.name,
                    chunk_size=config.chunk_size_words,
                    overlap=config.chunk_overlap_words,
                )
                st.session_state.chunks = chunks
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.index_built = False
                st.session_state.vector_namespace = None
            except PDFProcessingError as exc:
                st.error(f"PDF processing error: {exc}")

    if st.session_state.chunks and not st.session_state.index_built:
        try:
            with st.spinner("Building vector index…"):
                embedding_svc = EmbeddingService(
                    api_key=config.openai_api_key,
                    model=config.embedding_model,
                )
                texts = [chunk.text for chunk in st.session_state.chunks]
                embeddings = embedding_svc.embed_texts(texts)
                namespace = get_active_vector_namespace() or "default"
                vector_store = build_vector_store(config, index_name=st.session_state.cfg_index_name)

                lf = get_langfuse()
                if lf:
                    with lf.start_as_current_observation(
                        name="pdf_upload_and_index",
                        as_type="span",
                        input={"filename": uploaded_file.name if uploaded_file else namespace},
                    ):
                        vector_store.upsert_chunks(st.session_state.chunks, embeddings, namespace=namespace)
                        st.session_state.index_built = True
                        st.session_state.vector_namespace = namespace
                        try:
                            ns_stats = vector_store.describe_namespace(namespace)
                            lf.update_current_span(
                                metadata={
                                    "index_name": st.session_state.cfg_index_name,
                                    "namespace": namespace,
                                    "index_status": "ready",
                                    "chunks_in_index": ns_stats["namespace_vector_count"],
                                    "index_dimension": ns_stats["dimension"],
                                }
                            )
                        except VectorStoreError:
                            lf.update_current_span(
                                metadata={
                                    "index_name": st.session_state.cfg_index_name,
                                    "namespace": namespace,
                                    "index_status": "upserted",
                                    "chunks_in_index": len(st.session_state.chunks),
                                }
                            )
                else:
                    vector_store.upsert_chunks(st.session_state.chunks, embeddings, namespace=namespace)
                    st.session_state.index_built = True
                    st.session_state.vector_namespace = namespace

                # Phase 2: persist PDF to Supabase Storage when tenant is authenticated
                if st.session_state.get("tenant_authenticated") and st.session_state.get("tenant_id"):
                    try:
                        upload_pdf(
                            tenant_id=st.session_state.tenant_id,
                            filename=uploaded_file.name,
                            data=pdf_bytes,
                        )
                    except Exception:  # noqa: BLE001
                        pass  # Storage upload failure must not block indexing

                # Phase 4: advance status to INGESTED
                tid = st.session_state.get("tenant_id")
                if tid and st.session_state.get("tenant_status") == "DRAFT":
                    try:
                        update_tenant_status(tid, "INGESTED")
                        st.session_state["tenant_status"] = "INGESTED"
                        upsert_tenant_config(tid, {
                            "index_name": st.session_state.cfg_index_name,
                            "top_k": st.session_state.cfg_top_k,
                            "min_confidence": st.session_state.cfg_min_confidence,
                        })
                    except SupabaseError:
                        pass  # Status update failure must not block the UI

                st.rerun()

        except (EmbeddingServiceError, VectorStoreError, Exception) as exc:  # noqa: BLE001
            st.error(f"Index build error: {exc}")

    if st.session_state.chunks:
        st.info(f"📄 **{st.session_state.uploaded_file_name}** — {len(st.session_state.chunks)} chunks extracted")
        with st.expander("🔍 Index debug info", expanded=False):
            st.write(f"Index built: {st.session_state.index_built}")
            st.write(f"Namespace: {st.session_state.vector_namespace}")
            if st.session_state.index_built and st.session_state.vector_namespace:
                try:
                    namespace_stats = build_vector_store(config, index_name=st.session_state.cfg_index_name).describe_namespace(st.session_state.vector_namespace)
                    st.write("Index status: ready")
                    st.write(f"Chunks in index: {namespace_stats['namespace_vector_count']}")
                    st.write(f"Index dimension: {namespace_stats['dimension']}")
                except VectorStoreError as exc:
                    st.write(f"Index status: unavailable ({exc})")
            else:
                st.write("Index status: not built")
                st.write("Chunks in index: 0")


def render_evals_and_configs_tab(config) -> None:
    st.subheader("Evals & Configs")

    st.markdown("#### Retrieval settings")
    st.session_state.cfg_index_name = st.text_input(
        "Index name",
        value=st.session_state.cfg_index_name,
        key="evals_cfg_index_name",
    )
    st.session_state.cfg_top_k = st.number_input(
        "Top K",
        min_value=1,
        max_value=20,
        value=st.session_state.cfg_top_k,
        step=1,
        key="evals_cfg_top_k",
    )
    st.session_state.cfg_min_confidence = st.slider(
        "Min confidence",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.cfg_min_confidence,
        step=0.01,
        key="evals_cfg_min_confidence",
    )

    # Phase 4: persist updated config to Supabase whenever tenant is authenticated
    tid = st.session_state.get("tenant_id")
    if tid and st.session_state.get("tenant_authenticated"):
        if st.button("💾 Save retrieval settings", key="btn_save_cfg"):
            try:
                upsert_tenant_config(tid, {
                    "index_name": st.session_state.cfg_index_name,
                    "top_k": st.session_state.cfg_top_k,
                    "min_confidence": st.session_state.cfg_min_confidence,
                })
                st.success("Settings saved.")
            except SupabaseError as exc:
                st.error(f"Failed to save settings: {exc}")

    st.markdown("---")
    st.subheader("Evals")
    eval_file = st.file_uploader("Upload ground truth CSV", type=["csv"], key="eval_csv_uploader")
    if eval_file is None:
        st.session_state.eval_rows = None
        st.session_state.eval_file_name = None
    else:
        try:
            rows = parse_eval_csv(eval_file.getvalue())
            st.session_state.eval_rows = rows
            st.session_state.eval_file_name = eval_file.name
            st.success(f"Loaded {len(rows)} eval rows.")
            # Phase 2: persist CSV to Supabase Storage when tenant is authenticated
            if st.session_state.get("tenant_authenticated") and st.session_state.get("tenant_id"):
                try:
                    upload_eval_csv(
                        tenant_id=st.session_state.tenant_id,
                        filename=eval_file.name,
                        data=eval_file.getvalue(),
                    )
                except Exception:  # noqa: BLE001
                    pass  # Storage upload failure must not block eval loading
        except ValueError as exc:
            st.session_state.eval_rows = None
            st.session_state.eval_file_name = None
            st.error(str(exc))
        except UnicodeDecodeError:
            st.session_state.eval_rows = None
            st.session_state.eval_file_name = None
            st.error("Unable to read CSV. Please upload a UTF-8 encoded file.")
        except csv.Error as exc:
            st.session_state.eval_rows = None
            st.session_state.eval_file_name = None
            st.error(f"Unable to parse CSV: {exc}")

    eval_rows = st.session_state.eval_rows
    index_ready = st.session_state.index_built and st.session_state.vector_namespace
    can_run = bool(eval_rows) and bool(index_ready)

    if st.button("Run Evaluation", disabled=not can_run):
        config = load_config()
        total = len(eval_rows)
        results: list[dict] = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, row in enumerate(eval_rows, start=1):
            query = row["Query"]
            expected = row["Expected Response"]
            status_text.text(f"Evaluating query {i} of {total}…")
            try:
                actual = retrieve_and_answer(config, query)
            except (LLMServiceError, EmbeddingServiceError, VectorStoreError, Exception) as exc:  # noqa: BLE001
                actual = f"[Error during retrieval/generation: {exc}]"

            try:
                judgment = evaluate_response_with_judge(
                    openai_api_key=config.openai_api_key,
                    query=query,
                    expected_response=expected,
                    actual_response=actual,
                )
                score = judgment["score"]
                reason = judgment["reason"]
            except LLMServiceError as exc:
                score = 0
                reason = f"[Judge error: {exc}]"

            results.append(
                {
                    "Query": query,
                    "Expected Response": expected,
                    "Actual Response": actual,
                    "Score (1-5)": score,
                    "Judge Feedback": reason,
                }
            )
            progress_bar.progress(i / total)

        status_text.text("Evaluation complete.")
        st.session_state.eval_results = results

        # Phase 4: persist eval log + advance status to EVALUATED
        tid = st.session_state.get("tenant_id")
        if tid and st.session_state.get("tenant_status") == "INGESTED":
            try:
                save_eval_log(tid, [
                    {
                        "query": r["Query"],
                        "expected": r["Expected Response"],
                        "actual": r["Actual Response"],
                        "score": r["Score (1-5)"],
                        "reason": r["Judge Feedback"],
                    }
                    for r in results
                ])
                update_tenant_status(tid, "EVALUATED")
                st.session_state["tenant_status"] = "EVALUATED"
            except SupabaseError:
                pass  # Persistence failure must not block the eval display

    # --- Phase 3: Report display ---
    eval_results = st.session_state.eval_results
    if eval_results:
        valid_scores = [r["Score (1-5)"] for r in eval_results if isinstance(r["Score (1-5)"], int) and r["Score (1-5)"] > 0]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        st.metric("Average Score (1-5)", f"{avg_score:.2f}")

        st.dataframe(
            eval_results,
            column_order=["Query", "Expected Response", "Actual Response", "Score (1-5)", "Judge Feedback"],
            use_container_width=True,
        )

        csv_buffer = io.StringIO()
        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=["Query", "Expected Response", "Actual Response", "Score (1-5)", "Judge Feedback"],
        )
        writer.writeheader()
        writer.writerows(eval_results)
        st.download_button(
            label="⬇️ Download eval_report.csv",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="eval_report.csv",
            mime="text/csv",
        )
    # --- End Phase 3 ---

    # --- Phase 4 & 5: Publish button ---
    if st.session_state.get("tenant_status") == "EVALUATED" and st.session_state.get("tenant_id"):
        st.markdown("---")
        st.subheader("Publish Assistant")
        st.info("Your assistant has been evaluated. Publish it to generate a shareable URL.")
        if st.button("🚀 Publish", key="btn_publish"):
            from src.share import generate_share_token  # noqa: PLC0415
            tid = st.session_state.tenant_id
            cfg_share = load_config()
            jwt_secret = getattr(cfg_share, "jwt_secret", "") or ""
            if not jwt_secret:
                st.error("Set JWT_SECRET in your environment to enable shareable URLs.")
            else:
                try:
                    share_url = generate_share_token(
                        tenant_id=tid,
                        secret=jwt_secret,
                        base_url=getattr(cfg_share, "app_base_url", "") or "",
                    )
                    # Persist share token and advance status
                    from src.supabase_client import get_client as _get_client  # noqa: PLC0415
                    _c = _get_client()
                    if _c:
                        _c.table("tenants").update({"share_token": share_url}).eq("id", tid).execute()
                    update_tenant_status(tid, "PUBLISHED")
                    st.session_state["tenant_status"] = "PUBLISHED"
                    st.session_state["tenant_share_url"] = share_url
                    st.success("✅ Published!")
                    st.rerun()
                except SupabaseError as exc:
                    st.error(f"Publish failed: {exc}")

    if st.session_state.get("tenant_status") == "PUBLISHED" and st.session_state.get("tenant_share_url"):
        st.markdown("---")
        st.subheader("Share your Assistant")
        share_url = st.session_state.tenant_share_url
        st.text_input("Shareable URL", value=share_url, key="share_url_display", disabled=True)
        st.caption("Copy this URL and share it with your users.")


def render_chat_tab(config) -> None:
    if st.button("Clear chat"):
        clear_conversation()
        st.rerun()

    pending_question = st.session_state.pop("pending_question", None)
    if pending_question:
        process_chat_question(config, pending_question)

    last_message_index = len(st.session_state.messages) - 1
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                st.markdown("**Sources:**\n" + "\n".join(message["citations"]))
            if (
                index == last_message_index
                and message["role"] == "assistant"
                and st.session_state.last_retrieval_debug
            ):
                with st.expander("🧪 Retrieval debug info", expanded=False):
                    for label, value in st.session_state.last_retrieval_debug.items():
                        st.write(f"{label}: {value}")

            # --- Phase 5: thumbs-up / thumbs-down feedback ---
            if message["role"] == "assistant":
                _render_feedback_buttons(index, message.get("trace_id"))
            # --- End Phase 5 feedback ---


def _render_supabase_indicator() -> None:
    """Inject a colour dot indicating Supabase connection status into the sidebar.

    Previous approaches attempted to use ``position:fixed`` via st.markdown
    (broken because Streamlit's block container establishes a new stacking
    context that traps fixed elements) and then via st.components.v1.html with
    ``window.parent.document`` DOM manipulation (broken in deployed environments
    because component iframes are cross-origin, so accessing window.parent.document
    raises a SecurityError that silently swallows the operation).

    The reliable solution is to render the indicator inside st.sidebar using
    st.sidebar.markdown.  Sidebar HTML is emitted into the main Streamlit
    document (not inside a component iframe), so it is fully visible and
    correctly styled.
    """
    status, message = get_supabase_status()
    colour_map = {"grey": "#9e9e9e", "red": "#e53935", "green": "#43a047"}
    label_map = {"grey": "DB: not configured", "red": "DB: error", "green": "DB: connected"}
    colour = colour_map[status]
    label = label_map[status]
    html = (
        f'<div title="{message}" style="'
        "display:flex;align-items:center;gap:0.4rem;"
        "background:rgba(0,0,0,0.04);border-radius:999px;"
        "padding:0.15rem 0.65rem 0.15rem 0.4rem;"
        "font-size:0.75rem;font-weight:500;color:#333;"
        'margin-bottom:0.5rem;">'
        f'<span style="width:10px;height:10px;border-radius:50%;'
        f'background:{colour};flex-shrink:0;display:inline-block;"></span>'
        f"{label}"
        "</div>"
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)


def _render_tenant_sidebar_badge() -> None:
    """Show the current tenant name and lifecycle status badge in the sidebar."""
    tenant_name = st.session_state.get("tenant_name")
    tenant_status = st.session_state.get("tenant_status")
    if not tenant_name:
        return

    status_colour = {
        "DRAFT": "#9e9e9e",
        "INGESTED": "#1e88e5",
        "EVALUATED": "#fb8c00",
        "PUBLISHED": "#43a047",
    }
    colour = status_colour.get(str(tenant_status), "#9e9e9e") if tenant_status else "#9e9e9e"
    label = str(tenant_status) if tenant_status else "—"
    html = (
        f'<div style="'
        "display:flex;align-items:center;gap:0.4rem;"
        "background:rgba(0,0,0,0.04);border-radius:999px;"
        "padding:0.15rem 0.65rem 0.15rem 0.4rem;"
        "font-size:0.75rem;font-weight:500;color:#333;"
        'margin-bottom:0.5rem;">'
        f'<span style="width:10px;height:10px;border-radius:50%;'
        f'background:{colour};flex-shrink:0;display:inline-block;"></span>'
        f"Tenant: {tenant_name} · {label}"
        "</div>"
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="AI Assistant Builder", page_icon="🤖", layout="centered")
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    # --- Token routing: check for a shared assistant URL BEFORE any other UI ---
    token = st.query_params.get("token")
    if token:
        from src.share import verify_share_token, ShareTokenError  # noqa: PLC0415
        config = load_config()
        jwt_secret = getattr(config, "jwt_secret", "") or ""
        if not jwt_secret:
            st.error("This link cannot be verified: JWT_SECRET is not configured.")
            return
        try:
            tenant_id = verify_share_token(token, jwt_secret)
        except ShareTokenError:
            st.error("This link is invalid or has expired.")
            return
        # Confirm tenant is PUBLISHED before rendering the standalone chat view
        try:
            init_supabase(url=config.supabase_url, key=config.supabase_key)
            row = get_tenant(tenant_id)
        except SupabaseError:
            row = None
        if row is None or row.get("status") != "PUBLISHED":
            st.error("This assistant is not available.")
            return
        # --- Standalone chat view — no tab bar, no login, no other UI ---
        st.title(f"💬 {row.get('name', 'Assistant')} Chat")
        initialize_state()
        if not st.session_state.get("tenant_id"):
            _load_tenant_into_session(row)
        init_langfuse(
            secret_key=config.langfuse_secret_key,
            public_key=config.langfuse_public_key,
            host=config.langfuse_host,
        )
        kb_indexed = st.session_state.index_built and bool(st.session_state.vector_namespace)
        pending_question = st.session_state.pop("pending_question", None)
        if pending_question:
            process_chat_question(config, pending_question)
        for index, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("citations"):
                    st.markdown("**Sources:**\n" + "\n".join(message["citations"]))
                if message["role"] == "assistant":
                    _render_feedback_buttons(index, message.get("trace_id"))
        st.chat_input(
            "Ask anything...",
            key="chat_input_value",
            on_submit=queue_chat_question,
        )
        return
    # --- End token routing ---

    st.title("AI Assistant Builder")

    config = load_config()
    initialize_state()

    # Phase 1.5: probe Supabase connection and render the status indicator
    init_supabase(url=config.supabase_url, key=config.supabase_key)
    _render_supabase_indicator()
    _render_tenant_sidebar_badge()

    # Phase 4: initialise Langfuse once per session (no-op if keys absent)
    init_langfuse(
        secret_key=config.langfuse_secret_key,
        public_key=config.langfuse_public_key,
        host=config.langfuse_host,
    )

    if not config.openai_api_key:
        st.warning("Set OPENAI_API_KEY to start chatting.")
        return

    kb_indexed = st.session_state.index_built and bool(st.session_state.vector_namespace)
    tenant_authed = st.session_state.get("tenant_authenticated", False)

    tenant_tab, kb_tab, chat_tab, evals_tab = st.tabs(
        ["Tenant", "Knowledge Base", "Assistant Chat", "Evals & Configs"]
    )
    with tenant_tab:
        render_tenant_tab()
    with kb_tab:
        if not tenant_authed:
            st.info("🔒 Select or create a tenant in the **Tenant** tab to unlock this section.")
        else:
            render_knowledge_base_tab(config)
    with chat_tab:
        if not tenant_authed:
            st.info("🔒 Select or create a tenant in the **Tenant** tab to unlock chat.")
        elif not kb_indexed:
            st.info("🔒 Upload and index a PDF in the **Knowledge Base** tab to unlock chat.")
        else:
            render_chat_tab(config)
        # st.chat_input must be the last statement in this tab's scope,
        # not nested inside any container/form/columns, so it docks to the bottom.
        st.chat_input(
            "Ask anything...",
            key="chat_input_value",
            on_submit=queue_chat_question,
            disabled=not kb_indexed,
        )
    with evals_tab:
        if not tenant_authed:
            st.info("🔒 Select or create a tenant in the **Tenant** tab to unlock evaluations.")
        elif not kb_indexed:
            st.info("🔒 Upload and index a PDF in the **Knowledge Base** tab to unlock evaluations.")
        else:
            render_evals_and_configs_tab(config)


if __name__ == "__main__":
    main()
