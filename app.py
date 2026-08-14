from __future__ import annotations

import hashlib

import streamlit as st

from src.config import AppConfig, load_config
from src.embeddings import EmbeddingService, EmbeddingServiceError
from src.llm import LLMService, LLMServiceError
from src.pdf_processor import PDFProcessingError, build_chunks, extract_pdf_pages
from src.retrieval import format_citation, format_metrics_for_display
from src.vector_store import PineconeVectorStore, VectorStoreError


def initialize_state() -> None:
    defaults = {
        "messages": [],
        "document_id": None,
        "document_name": None,
        "namespace": None,
        "chunk_count": 0,
        "latest_answer": None,
        "latest_sources": [],
        "latest_metrics": None,
        "last_ingested_index": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.latest_answer = None
    st.session_state.latest_sources = []
    st.session_state.latest_metrics = None


def ingest_pdf(
    uploaded_file,
    config: AppConfig,
    embedding_service: EmbeddingService,
    vector_store: PineconeVectorStore,
) -> None:
    pdf_bytes = uploaded_file.getvalue()
    document_hash = hashlib.sha256(pdf_bytes).hexdigest()
    namespace = document_hash[:24]

    if (
        st.session_state.document_id == document_hash
        and st.session_state.last_ingested_index == config.pinecone_index_name
    ):
        return

    pages = extract_pdf_pages(pdf_bytes)
    chunks = build_chunks(
        pages,
        source=uploaded_file.name,
        chunk_size=config.chunk_size_words,
        overlap=config.chunk_overlap_words,
    )
    embeddings = embedding_service.embed_texts([chunk.text for chunk in chunks])
    vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings, namespace=namespace)

    st.session_state.document_id = document_hash
    st.session_state.document_name = uploaded_file.name
    st.session_state.namespace = namespace
    st.session_state.chunk_count = len(chunks)
    st.session_state.last_ingested_index = config.pinecone_index_name
    clear_conversation()


def render_sidebar(config: AppConfig) -> tuple[AppConfig, object]:
    st.sidebar.header("Configuration")
    pinecone_index_name = st.sidebar.text_input(
        "Index name",
        value=config.pinecone_index_name,
        help="Pinecone serverless index to create or reuse.",
    ).strip()
    top_k = st.sidebar.number_input(
        "TOP_K",
        min_value=1,
        max_value=10,
        value=config.top_k,
        step=1,
    )
    min_confidence_score = st.sidebar.slider(
        "Minimum confidence score",
        min_value=0.0,
        max_value=1.0,
        value=float(config.min_confidence_score),
        step=0.01,
    )
    if st.sidebar.button("Clear conversation"):
        clear_conversation()

    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    return config.with_overrides(
        pinecone_index_name=pinecone_index_name or config.pinecone_index_name,
        top_k=int(top_k),
        min_confidence_score=float(min_confidence_score),
    ), uploaded_file


def render_uploaded_document() -> None:
    st.subheader("Uploaded document")
    if st.session_state.document_name:
        st.success(
            f"Indexed {st.session_state.document_name} "
            f"({st.session_state.chunk_count} chunks)"
        )
    else:
        st.info("Upload a PDF to extract, chunk, embed, and index it.")


def render_chat() -> None:
    st.subheader("Chat interface")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_latest_details() -> None:
    st.subheader("Answer")
    if st.session_state.latest_answer:
        st.markdown(st.session_state.latest_answer)
    else:
        st.write("Ask a question about the uploaded PDF.")

    st.subheader("Sources / citations")
    if st.session_state.latest_sources:
        for source in st.session_state.latest_sources:
            st.markdown(f"- {format_citation(source)}")
    else:
        st.write("No sources yet.")

    st.subheader("Retrieval metrics")
    if st.session_state.latest_metrics:
        metrics = st.session_state.latest_metrics
        for label, value in format_metrics_for_display(metrics).items():
            st.write(f"**{label}:** {value}")
    else:
        st.write("No retrieval metrics yet.")


def build_embedding_service(config: AppConfig) -> EmbeddingService:
    return EmbeddingService(api_key=config.openai_api_key, model=config.embedding_model)


def build_llm_service(config: AppConfig) -> LLMService:
    return LLMService(api_key=config.openai_api_key, model=config.llm_model)


def build_vector_store(config: AppConfig) -> PineconeVectorStore:
    vector_store = PineconeVectorStore(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index_name,
        dimension=config.embedding_dimension,
        cloud=config.pinecone_cloud,
        region=config.pinecone_region,
    )
    return vector_store


def main() -> None:
    st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")
    st.title("PDF RAG Chatbot")
    initialize_state()

    config = load_config()
    config, uploaded_file = render_sidebar(config)

    missing_keys = []
    if not config.openai_api_key:
        missing_keys.append("OPENAI_API_KEY")
    if not config.pinecone_api_key:
        missing_keys.append("PINECONE_API_KEY")
    if missing_keys:
        st.warning(
            "Missing required environment variables: "
            + ", ".join(missing_keys)
            + ". Add them to your local .env file before indexing or asking questions."
        )

    render_uploaded_document()
    render_chat()
    render_latest_details()

    if uploaded_file is not None:
        if missing_keys:
            st.info("Upload received, but indexing is disabled until required API keys are configured.")
        else:
            try:
                embedding_service = build_embedding_service(config)
                vector_store = build_vector_store(config)
                with st.spinner("Extracting, chunking, embedding, and indexing the PDF..."):
                    ingest_pdf(uploaded_file, config, embedding_service, vector_store)
                st.success(f"{uploaded_file.name} is ready for questions.")
            except (PDFProcessingError, EmbeddingServiceError, VectorStoreError) as exc:
                st.error(str(exc))
                return

    question = st.chat_input("Ask a question about the uploaded PDF")
    if question is None:
        return

    question = question.strip()
    if not question:
        st.warning("Please enter a non-empty question.")
        return

    if not st.session_state.namespace:
        st.warning("Upload and index a PDF before asking questions.")
        return

    if missing_keys:
        st.warning("Configure the required API keys before asking questions.")
        return

    try:
        embedding_service = build_embedding_service(config)
        llm_service = build_llm_service(config)
        vector_store = build_vector_store(config)
        with st.spinner("Retrieving relevant chunks..."):
            question_embedding = embedding_service.embed_query(question)
            retrieval_response = vector_store.query(
                embedding=question_embedding,
                namespace=st.session_state.namespace,
                top_k=config.top_k,
                min_confidence_score=config.min_confidence_score,
            )

        if retrieval_response["relevant_count"] == 0:
            answer = (
                "I could not find enough relevant information in the uploaded document "
                "to answer that question."
            )
        else:
            with st.spinner("Generating answer from retrieved context..."):
                answer = llm_service.generate_answer(
                    question=question,
                    retrieval_results=retrieval_response["relevant_matches"],
                    conversation_history=st.session_state.messages,
                )

        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.latest_answer = answer
        st.session_state.latest_sources = retrieval_response["relevant_matches"]
        st.session_state.latest_metrics = retrieval_response["metrics"]
        st.rerun()
    except (EmbeddingServiceError, VectorStoreError, LLMServiceError) as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
