from __future__ import annotations

import hashlib

import streamlit as st

from src.config import AppConfig, load_config
from src.embeddings import EmbeddingService, EmbeddingServiceError
from src.llm import LLMService, LLMServiceError
from src.pdf_processor import PDFProcessingError, build_chunks, extract_pdf_pages
from src.retrieval import format_metrics_for_display
from src.vector_store import PineconeVectorStore, VectorStoreError

# ── Sentinel prefix used to detect "no relevant info" answers ─────────────────
_NO_INFO_PREFIX = "I could not find enough relevant information"

# ─────────────────────────────────────────────────────────────────────────────
# CSS / JS injection
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_CSS = """
<style>
/* ── Reset Streamlit chrome ───────────────────────────────────────────────── */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stAppViewContainer"] > section:first-child { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
.stMainBlockContainer { max-width: 100% !important; padding: 0 !important; }

/* ── Root vars ────────────────────────────────────────────────────────────── */
:root {
  --bg:          #f8f9fb;
  --surface:     #ffffff;
  --border:      #e2e4e9;
  --text:        #1a1d23;
  --text-muted:  #6b7280;
  --accent:      #4f6ef7;
  --accent-light:#eef0fd;
  --danger:      #dc2626;
  --warn-bg:     #fef9ec;
  --warn-border: #f59e0b;
  --radius:      10px;
  --header-h:    56px;
  --font: "Inter", "Segoe UI", system-ui, sans-serif;
}

* { box-sizing: border-box; }
body, .stApp { background: var(--bg) !important; font-family: var(--font); }

/* ── Push Streamlit content below fixed header ───────────────────────────── */
.stMainBlockContainer {
  padding-top: 20px !important;
  padding-bottom: 120px !important;
  max-width: 820px !important;
  margin: 0 auto !important;
  padding-left: 20px !important;
  padding-right: 20px !important;
}

/* ── Chat thread ──────────────────────────────────────────────────────────── */
.chat-thread { display: flex; flex-direction: column; gap: 16px; }

/* ── Message rows ─────────────────────────────────────────────────────────── */
.msg-row { display: flex; }
.msg-row.user  { justify-content: flex-end; }
.msg-row.assistant { justify-content: flex-start; align-items: flex-start; gap: 8px; }
.msg-bubble {
  max-width: 72%; padding: 12px 16px;
  border-radius: var(--radius);
  font-size: 14px; line-height: 1.6; word-break: break-word;
}
.msg-row.user .msg-bubble {
  background: var(--accent); color: #fff;
  border-bottom-right-radius: 3px;
}
.msg-row.assistant .msg-bubble {
  background: var(--surface); color: var(--text);
  border: 1px solid var(--border);
  border-bottom-left-radius: 3px;
  flex: 1; max-width: 100%;
}
.msg-bubble.no-info {
  background: var(--warn-bg); border-color: var(--warn-border); color: #92400e;
}

/* ── Citation chips ───────────────────────────────────────────────────────── */
.citations-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.citation-chip {
  position: relative;
  display: inline-flex; align-items: center; gap: 4px;
  background: var(--accent-light); color: var(--accent);
  border: 1px solid #c7d0fb; border-radius: 99px;
  padding: 3px 10px; font-size: 11px; font-weight: 600;
  cursor: default; white-space: nowrap;
  transition: background .15s, color .15s;
}
.citation-chip:hover { background: var(--accent); color: #fff; border-color: var(--accent); }

/* Hover tooltip via CSS ::after (no JS needed) */
.citation-chip[data-tooltip]:hover::after {
  content: attr(data-tooltip);
  position: absolute; bottom: calc(100% + 8px); left: 50%;
  transform: translateX(-50%);
  background: #1a1d23; color: #fff;
  padding: 8px 12px; border-radius: 8px;
  font-size: 12px; line-height: 1.5; font-weight: 400;
  white-space: pre-wrap; max-width: 300px; min-width: 160px;
  box-shadow: 0 4px 16px rgba(0,0,0,.2);
  z-index: 500; pointer-events: none;
  text-align: left;
}
.citation-chip[data-tooltip]:hover::before {
  content: '';
  position: absolute; bottom: calc(100% + 2px); left: 50%;
  transform: translateX(-50%);
  border: 6px solid transparent; border-top-color: #1a1d23;
  z-index: 500; pointer-events: none;
}

/* ── Info icon & metrics popover ──────────────────────────────────────────── */
.msg-info-wrap {
  position: relative;
  display: inline-flex; align-items: flex-start;
  margin-top: 2px; flex-shrink: 0;
}
.msg-info-btn {
  background: none; border: 1px solid var(--border);
  border-radius: 50%; width: 22px; height: 22px;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; font-size: 11px; color: var(--text-muted);
  flex-shrink: 0; line-height: 1;
  transition: background .15s, color .15s, border-color .15s;
}
.msg-info-btn:hover { background: var(--accent-light); color: var(--accent); border-color: var(--accent); }
.metrics-popover {
  display: none;
  position: absolute; bottom: calc(100% + 8px); right: 0;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 16px;
  min-width: 240px; max-width: 300px;
  box-shadow: 0 8px 32px rgba(0,0,0,.12);
  z-index: 300; font-size: 12px; color: var(--text);
}
.metrics-popover.open { display: block; }
.metrics-popover-title {
  font-size: 11px; font-weight: 700; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: .5px; margin-bottom: 10px;
}
.metrics-row {
  display: flex; justify-content: space-between;
  padding: 4px 0; border-bottom: 1px solid var(--border);
}
.metrics-row:last-child { border-bottom: none; }
.metrics-label { color: var(--text-muted); }
.metrics-value  { font-weight: 600; color: var(--text); }

/* ── Empty state ──────────────────────────────────────────────────────────── */
.empty-state {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 80px 40px; gap: 16px;
  color: var(--text-muted); text-align: center;
  min-height: 40vh;
}
.empty-state-icon    { font-size: 48px; opacity: .4; }
.empty-state-heading { font-size: 20px; font-weight: 700; color: var(--text); }
.empty-state-sub     { font-size: 14px; max-width: 340px; line-height: 1.6; }
.suggestion-chips    { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 12px; }
.suggestion-chip {
  background: var(--surface); border: 1px solid var(--border); border-radius: 99px;
  padding: 8px 18px; font-size: 13px; cursor: pointer; color: var(--text);
  transition: background .15s, border-color .15s, color .15s;
}
.suggestion-chip:hover { background: var(--accent-light); border-color: var(--accent); color: var(--accent); }

/* ── File attachment chip ─────────────────────────────────────────────────── */
.file-chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.file-chip {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--accent-light); color: var(--accent);
  border: 1px solid #c7d0fb; border-radius: 8px;
  padding: 6px 12px; font-size: 12px; font-weight: 600;
}
.file-chip-name { max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-chip-size { color: var(--text-muted); font-weight: 400; }
.file-chip-remove {
  background: none; border: none; cursor: pointer;
  color: var(--text-muted); font-size: 14px; line-height: 1;
  padding: 0 0 0 4px;
  transition: color .15s;
}
.file-chip-remove:hover { color: var(--danger); }

/* ── Typing indicator ─────────────────────────────────────────────────────── */
.typing-indicator {
  display: flex; align-items: center; gap: 5px;
  padding: 10px 14px; background: var(--surface);
  border: 1px solid var(--border); border-radius: var(--radius);
  width: fit-content;
}
.typing-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--text-muted);
  animation: typing-bounce 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: .2s; }
.typing-dot:nth-child(3) { animation-delay: .4s; }
@keyframes typing-bounce {
  0%,80%,100% { transform: translateY(0); opacity:.4; }
  40%          { transform: translateY(-6px); opacity:1; }
}

/* ── Compact file uploader (hide dropzone text, keep Browse button) ───────── */
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
[data-testid="stFileUploadDropzone"] {
  border: 1px dashed var(--border) !important;
  border-radius: 8px !important;
  padding: 8px 12px !important;
  background: transparent !important;
  min-height: auto !important;
}

/* ── Misc banners ─────────────────────────────────────────────────────────── */
.warn-banner {
  background: var(--warn-bg); border: 1px solid var(--warn-border);
  border-radius: var(--radius); padding: 8px 14px;
  font-size: 12px; color: #92400e; margin-bottom: 12px;
}
</style>
"""

_CHAT_JS = """
<script>
(function() {
  function init() {
    // ── Info icon → toggle metrics popover ────────────────────────────────────
    document.querySelectorAll('.msg-info-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        e.stopPropagation();
        const popover = btn.nextElementSibling;
        if (!popover) return;
        const wasOpen = popover.classList.contains('open');
        document.querySelectorAll('.metrics-popover.open')
          .forEach(p => p.classList.remove('open'));
        if (!wasOpen) popover.classList.add('open');
      });
    });

    // Close popovers on outside click
    document.addEventListener('click', e => {
      if (!e.target.closest('.msg-info-wrap')) {
        document.querySelectorAll('.metrics-popover.open')
          .forEach(p => p.classList.remove('open'));
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
    setTimeout(init, 400);
  }
})();
</script>
"""


# ─────────────────────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────────────────────

def initialize_state(config: AppConfig) -> None:
    defaults: dict = {
        "messages": [],
        "document_id": None,
        "document_name": None,
        "namespace": None,
        "chunk_count": 0,
        "last_ingested_index": None,
        "pending_suggestion": None,

    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_conversation() -> None:
    st.session_state.messages = []


def is_chat_ready(state: dict | None = None) -> bool:
    """A document is ready to chat once it is uploaded and indexed."""
    if state is None:
        state = st.session_state
    namespace = state.get("namespace")
    doc_name = state.get("document_name")
    chunk_count = state.get("chunk_count", 0)
    return bool(namespace and doc_name and chunk_count > 0)


def _remove_document() -> None:
    st.session_state.document_id = None
    st.session_state.document_name = None
    st.session_state.namespace = None
    st.session_state.chunk_count = 0
    st.session_state.last_ingested_index = None
    clear_conversation()


# ─────────────────────────────────────────────────────────────────────────────
# PDF ingestion (backend unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_pdf(
    uploaded_file,
    config: AppConfig,
    embedding_service: EmbeddingService,
    vector_store: "PineconeVectorStore",
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


# ─────────────────────────────────────────────────────────────────────────────
# Service factories (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def build_embedding_service(config: AppConfig) -> EmbeddingService:
    return EmbeddingService(api_key=config.openai_api_key, model=config.embedding_model)


def build_llm_service(config: AppConfig) -> LLMService:
    return LLMService(api_key=config.openai_api_key, model=config.llm_model)


def build_vector_store(config: AppConfig) -> PineconeVectorStore:
    return PineconeVectorStore(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index_name,
        dimension=config.embedding_dimension,
        cloud=config.pinecone_cloud,
        region=config.pinecone_region,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    """Minimal HTML-escape for injected text."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
    )


def _file_size_label(uploaded_file) -> str:
    try:
        size = len(uploaded_file.getvalue())
        if size >= 1024 * 1024:
            return f"{size / 1024 / 1024:.1f} MB"
        return f"{size / 1024:.0f} KB"
    except Exception:
        return ""


def _citation_chip_label(src) -> str:
    """Compact label: 'filename.pdf · p.5'."""
    if src.page is not None:
        return f"{src.source} · p.{src.page}"
    return f"{src.source} · chunk {src.chunk_index}"


def _build_citation_chips_html(sources: list) -> str:
    if not sources:
        return ""
    chips = ""
    for src in sources:
        label = _citation_chip_label(src)
        # Tooltip: first 200 chars of chunk text, newlines replaced with spaces
        tooltip = _esc(src.text[:200].replace("\n", " "))
        chips += (
            f'<span class="citation-chip" data-tooltip="{tooltip}">'
            f"📎 {_esc(label)}</span>"
        )
    return f'<div class="citations-row">{chips}</div>'


def _build_metrics_popover_html(metrics: dict) -> str:
    """Build the (i) button + hidden popover for one assistant message."""
    display = format_metrics_for_display(metrics)
    rows_data = [
        ("Retrieved",   display["Retrieved"]),
        ("Relevant",    display["Relevant"]),
        ("Threshold",   display["Threshold"]),
        ("Avg score",   display["Average score (retrieval-score-based precision proxy)"]),
        ("Recall proxy",display["Recall proxy (retrieval-score-based)"]),
    ]
    rows_html = "".join(
        f'<div class="metrics-row">'
        f'<span class="metrics-label">{_esc(lbl)}</span>'
        f'<span class="metrics-value">{_esc(val)}</span>'
        f'</div>'
        for lbl, val in rows_data
    )
    if metrics.get("scores"):
        scores_str = ", ".join(f"{s:.3f}" for s in metrics["scores"])
        rows_html += (
            f'<div class="metrics-row">'
            f'<span class="metrics-label">Scores</span>'
            f'<span class="metrics-value">{_esc(scores_str)}</span>'
            f'</div>'
        )
    return (
        '<div class="msg-info-wrap">'
        '<button class="msg-info-btn" title="Retrieval metrics">ℹ</button>'
        '<div class="metrics-popover">'
        '<div class="metrics-popover-title">Retrieval Metrics</div>'
        f'{rows_html}'
        '</div>'
        '</div>'
    )


def _build_message_html(
    role: str,
    content: str,
    sources: list | None = None,
    metrics: dict | None = None,
) -> str:
    is_no_info = role == "assistant" and content.startswith(_NO_INFO_PREFIX)
    bubble_cls = "no-info" if is_no_info else ""
    chips_html = _build_citation_chips_html(sources or []) if role == "assistant" else ""
    info_html = _build_metrics_popover_html(metrics) if (role == "assistant" and metrics) else ""

    if role == "assistant":
        return (
            f'<div class="msg-row assistant">'
            f'<div class="msg-bubble {bubble_cls}">'
            f'{_esc(content)}'
            f'{chips_html}'
            f'</div>'
            f'{info_html}'
            f'</div>'
        )
    return (
        f'<div class="msg-row user">'
        f'<div class="msg-bubble">{_esc(content)}</div>'
        f'</div>'
    )


def _build_file_chip_html(name: str, chunk_count: int) -> str:
    return (
        '<div class="file-chip-row">'
        '<div class="file-chip">'
        '<span>📄</span>'
        f'<span class="file-chip-name">{_esc(name)}</span>'
        f'<span class="file-chip-size">&nbsp;·&nbsp;{chunk_count} chunks</span>'
        '<button class="file-chip-remove" title="Remove document">✕</button>'
        '</div>'
        '</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")

    config = load_config()
    initialize_state(config)


    missing_keys: list[str] = []
    if not config.openai_api_key:
        missing_keys.append("OPENAI_API_KEY")
    if not config.pinecone_api_key:
        missing_keys.append("PINECONE_API_KEY")

    # ── Inject global CSS ────────────────────────────────────────────────────
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    # ── API key warning ──────────────────────────────────────────────────────
    if missing_keys:
        st.markdown(
            f'<div class="warn-banner">⚠️ Missing keys: {", ".join(missing_keys)}. '
            f"Add them to <code>.env</code> before indexing.</div>",
            unsafe_allow_html=True,
        )

    # ── Chat thread ──────────────────────────────────────────────────────────
    messages = st.session_state.messages

    # Detect whether we are waiting for an assistant reply (last message is user).
    _pending_response = bool(
        messages and messages[-1]["role"] == "user"
    )

    if messages:
        thread_html = '<div class="chat-thread">'
        for msg in messages:
            thread_html += _build_message_html(
                role=msg["role"],
                content=msg["content"],
                sources=msg.get("sources"),
                metrics=msg.get("metrics"),
            )
        # Show typing indicator while RAG pipeline is running
        if _pending_response:
            thread_html += (
                '<div class="msg-row assistant">'
                '<div class="typing-indicator">'
                '<div class="typing-dot"></div>'
                '<div class="typing-dot"></div>'
                '<div class="typing-dot"></div>'
                '</div>'
                '</div>'
            )
        thread_html += '</div>'
        st.markdown(thread_html, unsafe_allow_html=True)
        st.markdown(_CHAT_JS, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">💬</div>'
            '<div class="empty-state-heading">Ask about your document</div>'
            '<div class="empty-state-sub">'
            'Upload a PDF, then ask a question about it.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── File attachment chip + remove (when a doc is loaded) ─────────────────
    doc_name = st.session_state.document_name
    chunk_count = st.session_state.chunk_count

    if doc_name:
        st.markdown(_build_file_chip_html(doc_name, chunk_count), unsafe_allow_html=True)

    # ── File uploader (compact; appears just above chat input) ───────────────
    upload_label = "📎 Attach PDF"
    uploaded_file = st.file_uploader(
        upload_label,
        type=["pdf"],
        key="pdf_uploader",
        label_visibility="visible",
    )

    # ── Chat input ───────────────────────────────────────────────────────────
    source_ready = is_chat_ready()
    placeholder = (
        "Ask a question about the document…"
        if source_ready
        else "Attach a PDF first to start chatting…"
    )
    question = st.chat_input(placeholder, disabled=not source_ready, key="chat_input")

    # ── Handle file upload ───────────────────────────────────────────────────
    if uploaded_file is not None:
        if missing_keys:
            st.info("Upload received — indexing disabled until API keys are set.")
        else:
            try:
                embedding_service = build_embedding_service(config)
                vector_store = build_vector_store(config)
                with st.spinner("Indexing PDF…"):
                    ingest_pdf(uploaded_file, config, embedding_service, vector_store)
                st.rerun()
            except (PDFProcessingError, EmbeddingServiceError, VectorStoreError) as exc:
                st.error(str(exc))

    # ── Handle query ─────────────────────────────────────────────────────────
    if question is None and not _pending_response:
        return

    if not _pending_response:
        # First pass: validate, record the user message, then rerun so the
        # bubble appears immediately (with typing indicator) before the
        # slow RAG pipeline starts.
        question = question.strip()
        if not question:
            return

        if missing_keys:
            st.warning("Configure the required API keys before asking questions.")
            return

        st.session_state.messages.append({"role": "user", "content": question})
        st.rerun()

    # Second pass (rerun): last message is user → run RAG pipeline and append reply.
    pending_question = st.session_state.messages[-1]["content"]

    if missing_keys:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ Configure the required API keys before asking questions.",
            "sources": [],
            "metrics": None,
        })
        st.rerun()

    try:
        embedding_service = build_embedding_service(config)
        llm_service = build_llm_service(config)
        vector_store = build_vector_store(config)

        question_embedding = embedding_service.embed_query(pending_question)
        retrieval_response = vector_store.query(
            embedding=question_embedding,
            namespace=st.session_state.namespace,
            top_k=config.top_k,
            min_confidence_score=config.min_confidence_score,
        )

        if retrieval_response["relevant_count"] == 0:
            answer = _NO_INFO_PREFIX + " in the uploaded document to answer that question."
        else:
            answer = llm_service.generate_answer(
                question=pending_question,
                retrieval_results=retrieval_response["relevant_matches"],
                conversation_history=st.session_state.messages[:-1],
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": retrieval_response["relevant_matches"],
            "metrics": retrieval_response["metrics"],
        })
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ An error occurred: {exc}",
            "sources": [],
            "metrics": None,
        })
        st.rerun()


main()
