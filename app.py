from __future__ import annotations

import streamlit as st

from src.config import load_config
from src.llm import LLMService, LLMServiceError

_GLOBAL_CSS = """
<style>
.block-container { max-width: 900px; padding-top: 2rem; }
</style>
"""


def initialize_state() -> None:
    st.session_state.setdefault("messages", [])


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
    messages.append({"role": "assistant", "content": error_message})


def build_llm_service() -> LLMService:
    config = load_config()
    return LLMService(api_key=config.openai_api_key, model=config.llm_model)


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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        llm_service = build_llm_service()
        answer = llm_service.generate_answer(
            question=question,
            conversation_history=st.session_state.messages[:-1],
        )
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    except (LLMServiceError, Exception) as exc:  # noqa: BLE001
        show_chat_error(exc)


if __name__ == "__main__":
    main()
