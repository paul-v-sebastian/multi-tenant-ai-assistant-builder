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
from src.retrieval import format_citation
from src.vector_store import PineconeVectorStore, VectorStoreError

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
    # Phase 5 sidebar config defaults (POC values)
    st.session_state.setdefault("cfg_index_name", "my-pdf-index")
    st.session_state.setdefault("cfg_top_k", 3)
    st.session_state.setdefault("cfg_min_confidence", 0.80)
    st.session_state.setdefault("eval_rows", None)
    st.session_state.setdefault("eval_file_name", None)
    st.session_state.setdefault("eval_results", None)
    st.session_state.setdefault("last_retrieval_debug", None)
    st.session_state.setdefault("pending_question", None)


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

        message: dict = {"role": "assistant", "content": answer}
        if citations:
            message["citations"] = citations
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


def render_evals_tab() -> None:
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


def render_chat_tab(config) -> None:
    if st.button("Clear chat"):
        clear_conversation()
        st.rerun()

    # --- Phase 1: PDF upload and chunking / Phase 2: build index ---
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
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
                namespace = st.session_state.uploaded_file_name or "default"
                vector_store = build_vector_store(config, index_name=st.session_state.cfg_index_name)
                vector_store.upsert_chunks(st.session_state.chunks, embeddings, namespace=namespace)
                st.session_state.index_built = True
                st.session_state.vector_namespace = namespace
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
                    st.write(f"Index status: ready")
                    st.write(f"Chunks in index: {namespace_stats['namespace_vector_count']}")
                    st.write(f"Index dimension: {namespace_stats['dimension']}")
                except VectorStoreError as exc:
                    st.write(f"Index status: unavailable ({exc})")
            else:
                st.write("Index status: not built")
                st.write("Chunks in index: 0")
    # --- End Phase 1 / Phase 2 ---

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


def main() -> None:
    st.set_page_config(page_title="LLM Chat", page_icon="💬", layout="centered")
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    st.title("Simple LLM Chat")

    config = load_config()
    initialize_state()

    # --- Phase 5: Sidebar retrieval settings ---
    with st.sidebar:
        st.markdown("### Retrieval settings")
        st.session_state.cfg_index_name = st.text_input(
            "Index name",
            value=st.session_state.cfg_index_name,
        )
        st.session_state.cfg_top_k = st.number_input(
            "Top K",
            min_value=1,
            max_value=20,
            value=st.session_state.cfg_top_k,
            step=1,
        )
        st.session_state.cfg_min_confidence = st.slider(
            "Min confidence",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.cfg_min_confidence,
            step=0.01,
        )
    # --- End Phase 5 sidebar ---

    if not config.openai_api_key:
        st.warning("Set OPENAI_API_KEY to start chatting.")
        return

    chat_tab, evals_tab = st.tabs(["Chat", "Evals"])
    with chat_tab:
        render_chat_tab(config)
    with evals_tab:
        render_evals_tab()

    # Placed outside the tab blocks so Streamlit can dock it to the bottom of the viewport.
    # The on_submit callback only enqueues the question; it is consumed inside render_chat_tab.
    st.chat_input("Ask anything...", key="chat_input_value", on_submit=queue_chat_question)


if __name__ == "__main__":
    main()
