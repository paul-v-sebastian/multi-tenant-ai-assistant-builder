from __future__ import annotations

from html import escape
import json

import streamlit as st
import streamlit.components.v1 as components

from src.config import AppConfig, load_config
from src.embeddings import EmbeddingService, EmbeddingServiceError
from src.llm import LLMService, LLMServiceError
from src.retrieval import RetrievedChunk, format_citation
from src.vector_store import PineconeVectorStore, VectorStoreError

_GLOBAL_CSS = """
<style>
.block-container {
    max-width: 800px;
    padding-top: 1rem;
    padding-bottom: 8rem;
}
.chat-bubble-row {
    display: flex;
    width: 100%;
}
.chat-bubble {
    max-width: 82%;
    padding: 0.95rem 1rem;
    border-radius: 1rem;
    line-height: 1.55;
    border: 1px solid transparent;
}
.chat-bubble.user {
    margin-left: auto;
    background: rgba(37, 99, 235, 0.14);
    border-color: rgba(37, 99, 235, 0.2);
    font-weight: 600;
}
.chat-bubble.assistant {
    margin-right: auto;
    background: rgba(15, 23, 42, 0.06);
    border-color: rgba(15, 23, 42, 0.08);
}
div[data-testid="stChatInput"] {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    max-width: 800px;
    margin: 0 auto;
    padding: 0.75rem 1rem 1rem;
    background: rgba(255, 255, 255, 0.96);
    border-top: 1px solid rgba(15, 23, 42, 0.08);
    backdrop-filter: blur(10px);
    z-index: 1000;
}
div[data-testid="stChatInput"] > div {
    margin: 0;
}
[data-testid="stSidebar"] .stButton button {
    width: 100%;
}
@media (prefers-color-scheme: dark) {
    .chat-bubble.user {
        background: rgba(96, 165, 250, 0.18);
        border-color: rgba(96, 165, 250, 0.26);
    }
    .chat-bubble.assistant {
        background: rgba(148, 163, 184, 0.14);
        border-color: rgba(148, 163, 184, 0.18);
    }
    div[data-testid="stChatInput"] {
        background: rgba(9, 13, 20, 0.96);
        border-top-color: rgba(148, 163, 184, 0.18);
    }
}
</style>
"""


def initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("show_sources_by_default", False)


def clear_conversation() -> None:
    state = st.session_state
    if isinstance(state, dict):
        state["messages"] = []
    else:
        state.messages = []


def build_message(
    role: str,
    content: str,
    sources: list[RetrievedChunk] | None = None,
) -> dict:
    message = {"role": role, "content": content}
    if sources:
        message["sources"] = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "page": chunk.page,
                "score": chunk.score,
            }
            for chunk in sources
        ]
    return message


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
    messages.append(build_message("assistant", error_message))


def build_sidebar(config: AppConfig) -> None:
    with st.sidebar:
        st.header("Utilities")
        if st.button("Clear chat", use_container_width=True):
            clear_conversation()
            st.rerun()
        st.checkbox("Expand source details automatically", key="show_sources_by_default")
        if config.pinecone_api_key:
            st.caption("RAG source retrieval is enabled for responses.")
        else:
            st.caption("Add PINECONE_API_KEY to enable retrieved source chunks.")


def retrieve_relevant_chunks(question: str, config: AppConfig) -> list[RetrievedChunk]:
    if not config.pinecone_api_key:
        return []
    try:
        embedding_service = EmbeddingService(api_key=config.openai_api_key, model=config.embedding_model)
        vector_store = PineconeVectorStore(
            api_key=config.pinecone_api_key,
            index_name=config.pinecone_index_name,
            dimension=config.embedding_dimension,
            cloud=config.pinecone_cloud,
            region=config.pinecone_region,
        )
        results = vector_store.query(
            embedding=embedding_service.embed_query(question),
            namespace="",
            top_k=config.top_k,
            min_confidence_score=config.min_confidence_score,
        )
    except (EmbeddingServiceError, VectorStoreError, Exception) as exc:  # noqa: BLE001
        st.sidebar.caption(f"Source retrieval unavailable: {exc}")
        return []
    return results["relevant_matches"] or results["matches"]


def render_copy_button(text: str, key: str) -> None:
    button_html = f"""
    <html>
      <body style="margin:0;background:transparent;">
        <button
          id="{key}"
          title="Copy response"
          aria-label="Copy response"
          onclick='navigator.clipboard.writeText({json.dumps(text)}); this.innerText = "✅"; setTimeout(() => this.innerText = "📋", 1200);'
          style="border:1px solid rgba(148,163,184,0.4);border-radius:999px;background:transparent;padding:0.3rem 0.6rem;cursor:pointer;"
        >📋</button>
      </body>
    </html>
    """
    components.html(button_html, height=36)


def render_sources(sources: list[dict], expanded: bool) -> None:
    with st.expander("🔍 View Sources", expanded=expanded):
        for index, source in enumerate(sources):
            chunk = RetrievedChunk(
                chunk_id=source["chunk_id"],
                text=source["text"],
                source=source["source"],
                chunk_index=source["chunk_index"],
                page=source["page"],
                score=source["score"],
            )
            st.caption(f"{format_citation(chunk)} · score {chunk.score:.2f}")
            st.write(chunk.text)
            if index < len(sources) - 1:
                st.divider()


def render_message(message: dict, index: int) -> None:
    role = message["role"]
    content = escape(message["content"]).replace("\n", "<br>")
    with st.chat_message(role):
        st.markdown(
            f'<div class="chat-bubble-row"><div class="chat-bubble {role}">{content}</div></div>',
            unsafe_allow_html=True,
        )
        if role != "assistant":
            return
        action_cols = st.columns([1, 1.5, 6])
        with action_cols[0]:
            render_copy_button(message["content"], f"copy-{index}")
        with action_cols[1]:
            feedback = st.feedback("thumbs", key=f"feedback-{index}")
            if feedback is not None:
                message["feedback"] = "up" if feedback else "down"
        sources = message.get("sources") or []
        if sources:
            render_sources(sources, expanded=st.session_state.show_sources_by_default)


def build_llm_service() -> LLMService:
    config = load_config()
    return LLMService(api_key=config.openai_api_key, model=config.llm_model)


def main() -> None:
    st.set_page_config(page_title="LLM Chat", page_icon="💬", layout="centered")
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    st.title("Simple LLM Chat")

    config = load_config()
    initialize_state()
    build_sidebar(config)

    if not config.openai_api_key:
        st.warning("Set OPENAI_API_KEY to start chatting.")
        return

    for index, message in enumerate(st.session_state.messages):
        render_message(message, index)

    question = st.chat_input("Ask anything...")
    if not question:
        return

    st.session_state.messages.append(build_message("user", question))
    render_message(st.session_state.messages[-1], len(st.session_state.messages) - 1)

    try:
        retrieved_chunks = retrieve_relevant_chunks(question, config)
        llm_service = build_llm_service()
        answer = llm_service.generate_answer(
            question=question,
            conversation_history=st.session_state.messages[:-1],
            retrieved_chunks=retrieved_chunks,
        )
        st.session_state.messages.append(build_message("assistant", answer, sources=retrieved_chunks))
        render_message(st.session_state.messages[-1], len(st.session_state.messages) - 1)
    except (LLMServiceError, Exception) as exc:  # noqa: BLE001
        show_chat_error(exc)


if __name__ == "__main__":
    main()
