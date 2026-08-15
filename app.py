from __future__ import annotations

import streamlit as st

from src.config import load_config
from src.embeddings import EmbeddingService, EmbeddingServiceError
from src.llm import LLMService, LLMServiceError
from src.pdf_processor import PDFProcessingError, build_chunks, extract_pdf_pages
from src.retrieval import format_citation
from src.vector_store import PineconeVectorStore, VectorStoreError

_GLOBAL_CSS = """
<style>
.block-container { max-width: 900px; padding-top: 2rem; }
</style>
"""


def initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("uploaded_file_name", None)
    st.session_state.setdefault("index_built", False)
    st.session_state.setdefault("vector_namespace", None)


def clear_conversation() -> None:
    state = st.session_state
    if isinstance(state, dict):
        state["messages"] = []
    else:
        state.messages = []


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


def build_llm_service() -> LLMService:
    config = load_config()
    return LLMService(api_key=config.openai_api_key, model=config.llm_model)


def build_vector_store(config) -> PineconeVectorStore:
    return PineconeVectorStore(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index_name,
        dimension=config.embedding_dimension,
        cloud=config.pinecone_cloud,
        region=config.pinecone_region,
    )


def main() -> None:
    st.set_page_config(page_title="LLM Chat", page_icon="💬", layout="centered")
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    st.title("Simple LLM Chat")

    config = load_config()
    initialize_state()

    if not config.openai_api_key:
        st.warning("Set OPENAI_API_KEY to start chatting.")
        return

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
                vector_store = build_vector_store(config)
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
                    namespace_stats = build_vector_store(config).describe_namespace(st.session_state.vector_namespace)
                    st.write(f"Index status: ready")
                    st.write(f"Chunks in index: {namespace_stats['namespace_vector_count']}")
                    st.write(f"Index dimension: {namespace_stats['dimension']}")
                except VectorStoreError as exc:
                    st.write(f"Index status: unavailable ({exc})")
            else:
                st.write("Index status: not built")
                st.write("Chunks in index: 0")
    # --- End Phase 1 / Phase 2 ---

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                st.markdown("**Sources:**\n" + "\n".join(message["citations"]))

    question = st.chat_input("Ask anything...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        llm_service = build_llm_service()

        if st.session_state.index_built and st.session_state.vector_namespace:
            # --- Phase 3: retrieval-augmented answer ---
            embedding_svc = EmbeddingService(
                api_key=config.openai_api_key,
                model=config.embedding_model,
            )
            question_embedding = embedding_svc.embed_query(question)
            vector_store = build_vector_store(config)
            retrieval_result = vector_store.query(
                embedding=question_embedding,
                namespace=st.session_state.vector_namespace,
                top_k=config.top_k,
                min_confidence_score=config.min_confidence_score,
            )
            metrics = retrieval_result["metrics"]
            relevant_matches = retrieval_result["relevant_matches"]
            with st.expander("🧪 Retrieval debug info", expanded=False):
                st.write(f"Threshold: {metrics['threshold']:.2f}")
                st.write(f"Retrieved: {metrics['retrieved_count']}")
                st.write(f"Relevant: {metrics['relevant_count']}")
                st.write(f"Precision: {metrics['precision']:.2f}")
                st.write(f"Recall: {metrics['recall']:.2f}")
                st.write(
                    "Scores: "
                    + (", ".join(f"{score:.2f}" for score in metrics["scores"]) if metrics["scores"] else "None")
                )

            if not relevant_matches:
                answer = llm_service.generate_no_context_answer(
                    question=question,
                    conversation_history=st.session_state.messages[:-1],
                )
                citations = []
            else:
                context = "\n\n".join(
                    f"[Chunk {m.chunk_index}] {m.text}" for m in relevant_matches
                )
                answer = llm_service.generate_answer_with_context(
                    question=question,
                    context=context,
                    conversation_history=st.session_state.messages[:-1],
                )
                citations = [format_citation(m) for m in relevant_matches]
        else:
            # --- plain chat (no document uploaded / index not ready) ---
            answer = llm_service.generate_answer(
                question=question,
                conversation_history=st.session_state.messages[:-1],
            )
            citations = []

        message: dict = {"role": "assistant", "content": answer}
        if citations:
            message["citations"] = citations
        st.session_state.messages.append(message)
        with st.chat_message("assistant"):
            st.markdown(answer)
            if citations:
                st.markdown("**Sources:**\n" + "\n".join(citations))
    except (LLMServiceError, EmbeddingServiceError, VectorStoreError, Exception) as exc:  # noqa: BLE001
        show_chat_error(exc)


if __name__ == "__main__":
    main()
