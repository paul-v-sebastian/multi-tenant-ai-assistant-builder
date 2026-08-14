from __future__ import annotations

import hashlib

import streamlit as st

from src.config import AppConfig, load_config
from src.embeddings import EmbeddingService, EmbeddingServiceError
from src.llm import LLMService, LLMServiceError
from src.pdf_processor import PDFProcessingError, build_chunks, extract_pdf_pages
from src.retrieval import format_citation, format_metrics_for_display
from src.vector_store import PineconeVectorStore, VectorStoreError

# ── Sentinel prefix used to detect "no relevant info" answers ─────────────────
_NO_INFO_PREFIX = "I could not find enough relevant information"

# ── Suggested prompts shown in the empty chat state ──────────────────────────
_SUGGESTION_CHIPS = [
    "Summarize this document",
    "What are the key findings?",
    "What is the main topic?",
    "List the conclusions",
]


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

/* ── Root vars ────────────────────────────────────────────────────────────── */
:root {
  --bg:        #f8f9fb;
  --surface:   #ffffff;
  --border:    #e2e4e9;
  --text:      #1a1d23;
  --text-muted:#6b7280;
  --accent:    #4f6ef7;
  --accent-light: #eef0fd;
  --danger:    #dc2626;
  --warn-bg:   #fef9ec;
  --warn-border:#f59e0b;
  --radius:    10px;
  --pane-left-w:  280px;
  --pane-right-w: 300px;
  --pane-rail-w:  52px;
  --transition:   220ms ease;
  --font: "Inter", "Segoe UI", system-ui, sans-serif;
}

* { box-sizing: border-box; }

body, .stApp { background: var(--bg) !important; font-family: var(--font); }

/* ── 3-pane shell ─────────────────────────────────────────────────────────── */
.rag-shell {
  display: flex;
  height: 100vh;
  overflow: hidden;
  background: var(--bg);
}

/* ── Side pane shared ─────────────────────────────────────────────────────── */
.rag-pane {
  display: flex;
  flex-direction: column;
  background: var(--surface);
  border-right: 1px solid var(--border);
  transition: width var(--transition);
  overflow: hidden;
  flex-shrink: 0;
}
.rag-pane-right {
  border-right: none;
  border-left: 1px solid var(--border);
}
.rag-pane.collapsed { width: var(--pane-rail-w) !important; }
.rag-pane-left  { width: var(--pane-left-w); }
.rag-pane-right { width: var(--pane-right-w); }

.pane-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 14px 12px 10px;
  border-bottom: 1px solid var(--border);
  min-height: 52px;
  flex-shrink: 0;
}
.pane-header-icon { font-size: 18px; flex-shrink: 0; }
.pane-header-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: .3px;
  white-space: nowrap;
  flex: 1;
}
.pane-collapse-btn {
  background: none;
  border: none;
  cursor: pointer;
  color: var(--text-muted);
  font-size: 16px;
  padding: 2px 4px;
  border-radius: 4px;
  flex-shrink: 0;
  line-height: 1;
}
.pane-collapse-btn:hover { background: var(--accent-light); color: var(--accent); }
.pane-body {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 12px;
}
.collapsed .pane-header-title,
.collapsed .pane-body { display: none; }
.collapsed .pane-header { justify-content: center; padding: 14px 0 10px; border-bottom: 1px solid var(--border); }
.rail-badge {
  display: none;
  position: absolute;
  top: 6px; right: 4px;
  background: var(--accent);
  color: #fff;
  font-size: 9px;
  border-radius: 99px;
  padding: 1px 4px;
  font-weight: 700;
}
.collapsed .rail-badge { display: block; }
.pane-header { position: relative; }

/* ── Center pane ──────────────────────────────────────────────────────────── */
.rag-center {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-width: 0;
  background: var(--bg);
}
.center-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 14px 20px 10px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  min-height: 52px;
  flex-shrink: 0;
}
.center-header-title {
  font-size: 14px;
  font-weight: 700;
  color: var(--text);
  flex: 1;
}

/* ── Chat thread ──────────────────────────────────────────────────────────── */
.chat-thread {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.msg-row { display: flex; }
.msg-row.user { justify-content: flex-end; }
.msg-row.assistant { justify-content: flex-start; }
.msg-bubble {
  max-width: 72%;
  padding: 12px 16px;
  border-radius: var(--radius);
  font-size: 14px;
  line-height: 1.6;
  word-break: break-word;
}
.msg-row.user .msg-bubble {
  background: var(--accent);
  color: #fff;
  border-bottom-right-radius: 3px;
}
.msg-row.assistant .msg-bubble {
  background: var(--surface);
  color: var(--text);
  border: 1px solid var(--border);
  border-bottom-left-radius: 3px;
}
.msg-bubble.no-info {
  background: var(--warn-bg);
  border-color: var(--warn-border);
  color: #92400e;
}

/* ── Citation chips ───────────────────────────────────────────────────────── */
.citations-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.citation-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--accent-light);
  color: var(--accent);
  border: 1px solid #c7d0fb;
  border-radius: 99px;
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
}
.citation-chip:hover { background: var(--accent); color: #fff; }

/* ── Typing indicator ─────────────────────────────────────────────────────── */
.typing-indicator {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 10px 14px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  width: fit-content;
}
.typing-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--text-muted);
  animation: typing-bounce 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: .2s; }
.typing-dot:nth-child(3) { animation-delay: .4s; }
@keyframes typing-bounce {
  0%,80%,100% { transform: translateY(0); opacity:.4; }
  40%          { transform: translateY(-6px); opacity:1; }
}

/* ── Empty-state ──────────────────────────────────────────────────────────── */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  gap: 20px;
  color: var(--text-muted);
  text-align: center;
  padding: 40px;
}
.empty-state-icon { font-size: 48px; opacity: .4; }
.empty-state-heading { font-size: 18px; font-weight: 600; color: var(--text); }
.empty-state-sub { font-size: 13px; max-width: 340px; }
.suggestion-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 8px; }
.suggestion-chip {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 99px;
  padding: 6px 16px;
  font-size: 13px;
  cursor: pointer;
  transition: background .15s, border-color .15s;
  color: var(--text);
}
.suggestion-chip:hover { background: var(--accent-light); border-color: var(--accent); color: var(--accent); }

/* ── Chat input bar ───────────────────────────────────────────────────────── */
.chat-input-bar {
  padding: 14px 20px;
  border-top: 1px solid var(--border);
  background: var(--surface);
  flex-shrink: 0;
}

/* ── Source cards (left pane) ─────────────────────────────────────────────── */
.source-card {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 8px;
  cursor: pointer;
  transition: border-color .15s, background .15s;
  background: var(--surface);
}
.source-card.active { border-color: var(--accent); background: var(--accent-light); }
.source-card.highlighted { box-shadow: 0 0 0 2px var(--accent); }
.source-card-icon { font-size: 20px; flex-shrink: 0; margin-top: 1px; }
.source-card-body { flex: 1; min-width: 0; }
.source-card-name { font-size: 13px; font-weight: 600; color: var(--text); word-break: break-all; }
.source-card-meta { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
.status-dot.ready   { background: #22c55e; }
.status-dot.processing { background: #f59e0b; animation: pulse 1.5s infinite; }
.status-dot.error   { background: var(--danger); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
.source-card-check { flex-shrink: 0; margin-top: 3px; }

/* ── Stat cards (right pane) ──────────────────────────────────────────────── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }
.stat-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 12px;
}
.stat-card-value { font-size: 20px; font-weight: 700; color: var(--text); }
.stat-card-label { font-size: 10px; color: var(--text-muted); margin-top: 2px; text-transform: uppercase; letter-spacing: .5px; }
.scores-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.scores-table th, .scores-table td { padding: 5px 8px; text-align: left; border-bottom: 1px solid var(--border); }
.scores-table th { color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 10px; letter-spacing: .5px; }

/* ── Misc ─────────────────────────────────────────────────────────────────── */
.section-label {
  font-size: 11px;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: .6px;
  margin: 16px 0 8px;
}
.section-label:first-child { margin-top: 0; }
.divider { height: 1px; background: var(--border); margin: 14px 0; }
.warn-banner {
  background: var(--warn-bg);
  border: 1px solid var(--warn-border);
  border-radius: var(--radius);
  padding: 8px 12px;
  font-size: 12px;
  color: #92400e;
  margin-bottom: 12px;
}
.info-banner {
  background: var(--accent-light);
  border: 1px solid #c7d0fb;
  border-radius: var(--radius);
  padding: 8px 12px;
  font-size: 12px;
  color: #3730a3;
  margin-bottom: 12px;
}

/* ── Source overlay ───────────────────────────────────────────────────────── */
.source-overlay-backdrop {
  display: none;
  position: fixed; inset: 0;
  background: rgba(0,0,0,.45);
  z-index: 1000;
}
.source-overlay-backdrop.open { display: flex; align-items: center; justify-content: center; }
.source-overlay {
  background: var(--surface);
  border-radius: 14px;
  width: min(600px, 92vw);
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(0,0,0,.25);
}
.source-overlay-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
}
.source-overlay-title { font-size: 15px; font-weight: 700; flex: 1; color: var(--text); }
.source-overlay-close {
  background: none; border: none; cursor: pointer;
  font-size: 20px; color: var(--text-muted); padding: 0 4px;
}
.source-overlay-close:hover { color: var(--text); }
.source-overlay-body { flex: 1; overflow-y: auto; padding: 20px; font-size: 14px; line-height: 1.7; color: var(--text); white-space: pre-wrap; }

/* ── Mobile ───────────────────────────────────────────────────────────────── */
@media (max-width: 767px) {
  .rag-pane { position: fixed; top: 0; bottom: 0; z-index: 200; box-shadow: 4px 0 24px rgba(0,0,0,.15); }
  .rag-pane-left  { left: 0; transform: translateX(-100%); transition: transform var(--transition); }
  .rag-pane-right { right: 0; transform: translateX(100%); border-left: 1px solid var(--border); transition: transform var(--transition); }
  .rag-pane-left.mobile-open  { transform: translateX(0); }
  .rag-pane-right.mobile-open { transform: translateX(0); }
  .rag-pane.collapsed { width: var(--pane-left-w) !important; }
  .mobile-top-bar {
    display: flex !important;
  }
}
.mobile-top-bar {
  display: none;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
}
.mobile-icon-btn {
  background: none; border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 10px; cursor: pointer; font-size: 18px;
}
.mobile-overlay-bg {
  display: none;
  position: fixed; inset: 0; background: rgba(0,0,0,.3); z-index: 199;
}
.mobile-overlay-bg.open { display: block; }
</style>
"""

_COLLAPSE_JS = """
<script>
(function() {
  const KEYS = { left: 'rag_left_collapsed', right: 'rag_right_collapsed' };

  function applyState(side, collapsed) {
    const pane = document.querySelector('.rag-pane-' + side);
    if (!pane) return;
    if (collapsed) pane.classList.add('collapsed');
    else pane.classList.remove('collapsed');
    const btn = pane.querySelector('.pane-collapse-btn');
    if (btn) btn.textContent = collapsed ? (side === 'left' ? '›' : '‹') : (side === 'left' ? '‹' : '›');
  }

  function togglePane(side) {
    const cur = localStorage.getItem(KEYS[side]) === '1';
    const next = !cur;
    localStorage.setItem(KEYS[side], next ? '1' : '0');
    applyState(side, next);
  }

  function init() {
    applyState('left',  localStorage.getItem(KEYS.left)  === '1');
    applyState('right', localStorage.getItem(KEYS.right) === '1');

    document.querySelectorAll('[data-collapse]').forEach(btn => {
      btn.addEventListener('click', () => togglePane(btn.dataset.collapse));
    });

    // Suggestion chips → fill streamlit chat input and submit
    document.querySelectorAll('.suggestion-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const txt = chip.dataset.q;
        const inp = document.querySelector('[data-testid="stChatInputTextArea"]');
        if (inp) {
          const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
          nativeSetter.call(inp, txt);
          inp.dispatchEvent(new Event('input', { bubbles: true }));
          const form = inp.closest('form');
          if (form) {
            const submitBtn = form.querySelector('button[type="submit"], button[kind="primaryFormSubmit"]');
            if (submitBtn) submitBtn.click();
          }
        }
      });
    });

    // Citation chip → highlight source card + open overlay
    document.querySelectorAll('.citation-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const src = chip.dataset.source;
        const text = chip.dataset.text || '';

        // Highlight matching source card in left pane
        document.querySelectorAll('.source-card').forEach(c => c.classList.remove('highlighted'));
        const card = document.querySelector(`.source-card[data-name="${CSS.escape(src)}"]`);
        if (card) {
          card.classList.add('highlighted');
          card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          // Expand left pane if collapsed
          const leftPane = document.querySelector('.rag-pane-left');
          if (leftPane && leftPane.classList.contains('collapsed')) {
            localStorage.setItem('rag_left_collapsed', '0');
            applyState('left', false);
          }
        }

        // Open overlay
        const backdrop = document.getElementById('src-overlay-backdrop');
        const titleEl  = document.getElementById('src-overlay-title');
        const bodyEl   = document.getElementById('src-overlay-body');
        if (backdrop && titleEl && bodyEl) {
          titleEl.textContent = src;
          bodyEl.textContent  = text || '(No preview text available)';
          backdrop.classList.add('open');
        }
      });
    });

    document.getElementById('src-overlay-close')?.addEventListener('click', closeOverlay);
    document.getElementById('src-overlay-backdrop')?.addEventListener('click', e => {
      if (e.target.id === 'src-overlay-backdrop') closeOverlay();
    });
    function closeOverlay() {
      document.getElementById('src-overlay-backdrop')?.classList.remove('open');
      document.querySelectorAll('.source-card').forEach(c => c.classList.remove('highlighted'));
    }

    // Mobile drawer toggles
    document.getElementById('mob-left-btn')?.addEventListener('click', () => {
      document.querySelector('.rag-pane-left')?.classList.toggle('mobile-open');
      document.getElementById('mob-overlay')?.classList.toggle('open');
    });
    document.getElementById('mob-right-btn')?.addEventListener('click', () => {
      document.querySelector('.rag-pane-right')?.classList.toggle('mobile-open');
      document.getElementById('mob-overlay')?.classList.toggle('open');
    });
    document.getElementById('mob-overlay')?.addEventListener('click', () => {
      document.querySelector('.rag-pane-left')?.classList.remove('mobile-open');
      document.querySelector('.rag-pane-right')?.classList.remove('mobile-open');
      document.getElementById('mob-overlay')?.classList.remove('open');
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
    // Retry shortly after Streamlit re-renders
    setTimeout(init, 400);
  }
})();
</script>
"""


# ─────────────────────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────────────────────

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
        "source_active": True,
        "pending_suggestion": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.latest_answer = None
    st.session_state.latest_sources = []
    st.session_state.latest_metrics = None


# ─────────────────────────────────────────────────────────────────────────────
# PDF ingestion (backend unchanged)
# ─────────────────────────────────────────────────────────────────────────────

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


def _build_source_card_html(name: str, meta: str, status: str, active: bool) -> str:
    active_cls = "active" if active else ""
    return f"""
<div class="source-card {active_cls}" data-name="{_esc(name)}">
  <span class="source-card-icon">📄</span>
  <div class="source-card-body">
    <div class="source-card-name">{_esc(name)}</div>
    <div class="source-card-meta">
      <span class="status-dot {status}"></span>{_esc(meta)}
    </div>
  </div>
  <input class="source-card-check" type="checkbox" {"checked" if active else ""} readonly>
</div>
"""


def _build_citation_chips_html(sources: list) -> str:
    if not sources:
        return ""
    chips = ""
    for src in sources:
        label = format_citation(src)
        preview_text = _esc(src.text[:800])
        src_name = _esc(src.source)
        chips += (
            f'<span class="citation-chip" data-source="{src_name}" data-text="{preview_text}">'
            f"🔖 {_esc(label)}</span>"
        )
    return f'<div class="citations-row">{chips}</div>'


def _build_message_html(role: str, content: str, sources: list | None = None) -> str:
    is_no_info = content.startswith(_NO_INFO_PREFIX)
    extra_cls = "no-info" if (role == "assistant" and is_no_info) else ""
    chips = _build_citation_chips_html(sources or []) if role == "assistant" else ""
    return f"""
<div class="msg-row {_esc(role)}">
  <div class="msg-bubble {extra_cls}">
    {_esc(content)}
    {chips}
  </div>
</div>
"""


def _build_stat_card(value: str, label: str) -> str:
    return f"""
<div class="stat-card">
  <div class="stat-card-value">{_esc(value)}</div>
  <div class="stat-card-label">{_esc(label)}</div>
</div>
"""


def _build_scores_table(scores: list[float]) -> str:
    if not scores:
        return "<p style='font-size:12px;color:var(--text-muted)'>No scores</p>"
    rows = "".join(
        f"<tr><td>#{i+1}</td><td>{s:.4f}</td></tr>"
        for i, s in enumerate(scores)
    )
    return f"""
<table class="scores-table">
  <thead><tr><th>#</th><th>Score</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="RAG Notebook", page_icon="📓", layout="wide")
    initialize_state()

    config = load_config()

    missing_keys: list[str] = []
    if not config.openai_api_key:
        missing_keys.append("OPENAI_API_KEY")
    if not config.pinecone_api_key:
        missing_keys.append("PINECONE_API_KEY")

    # ── Inject global CSS ────────────────────────────────────────────────────
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    # ── Open shell ───────────────────────────────────────────────────────────
    st.markdown('<div class="rag-shell">', unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # LEFT PANE — Sources
    # ────────────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="rag-pane rag-pane-left" id="pane-left">
  <div class="pane-header">
    <span class="pane-header-icon">📁</span>
    <span class="pane-header-title">Sources</span>
    <button class="pane-collapse-btn" data-collapse="left" title="Collapse">‹</button>
    <span class="rail-badge" id="src-badge">0</span>
  </div>
  <div class="pane-body" id="pane-left-body">
""", unsafe_allow_html=True)

    # Warnings
    if missing_keys:
        st.markdown(
            f'<div class="warn-banner">⚠️ Missing keys: {", ".join(missing_keys)}. '
            f'Add them to <code>.env</code> before indexing.</div>',
            unsafe_allow_html=True,
        )

    # Source card or empty state
    doc_name = st.session_state.document_name
    if doc_name:
        # Show source card
        meta = f"{st.session_state.chunk_count} chunks"
        active = st.session_state.source_active
        st.markdown(
            _build_source_card_html(doc_name, meta, "ready", active),
            unsafe_allow_html=True,
        )
        # Toggle active checkbox via a real Streamlit checkbox (hidden beneath the card)
        new_active = st.checkbox(
            "Active",
            value=active,
            key="src_active_cb",
            label_visibility="collapsed",
        )
        if new_active != active:
            st.session_state.source_active = new_active
            st.rerun()

        st.markdown(
            '<div class="section-label" style="margin-top:14px">+ Add source</div>',
            unsafe_allow_html=True,
        )

    # File uploader (always present; acts as "Add source" CTA)
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    st.markdown('</div></div>', unsafe_allow_html=True)  # close pane-body + pane-left

    # ── Handle upload ────────────────────────────────────────────────────────
    if uploaded_file is not None:
        if missing_keys:
            st.markdown(
                '<div class="info-banner">Upload received — indexing disabled until API keys are set.</div>',
                unsafe_allow_html=True,
            )
        else:
            try:
                embedding_service = build_embedding_service(config)
                vector_store = build_vector_store(config)
                with st.spinner("Indexing PDF…"):
                    ingest_pdf(uploaded_file, config, embedding_service, vector_store)
                st.session_state.source_active = True
            except (PDFProcessingError, EmbeddingServiceError, VectorStoreError) as exc:
                st.error(str(exc))

    # ────────────────────────────────────────────────────────────────────────
    # CENTER PANE — Chat
    # ────────────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="rag-center">
  <div class="center-header">
    <span style="font-size:18px">💬</span>
    <span class="center-header-title">Chat</span>
  </div>
""", unsafe_allow_html=True)

    messages = st.session_state.messages
    latest_sources = st.session_state.latest_sources

    if messages:
        st.markdown('<div class="chat-thread" id="chat-thread">', unsafe_allow_html=True)
        # Pair messages; attach sources to last assistant message
        for i, msg in enumerate(messages):
            is_last_assistant = (
                msg["role"] == "assistant" and i == len(messages) - 1
            )
            srcs = latest_sources if is_last_assistant else []
            st.markdown(_build_message_html(msg["role"], msg["content"], srcs), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Empty state with suggestion chips
        suggestions_html = "".join(
            f'<button class="suggestion-chip" data-q="{_esc(q)}">{_esc(q)}</button>'
            for q in _SUGGESTION_CHIPS
        )
        st.markdown(f"""
<div class="chat-thread" style="display:flex;flex-direction:column;">
  <div class="empty-state">
    <div class="empty-state-icon">📓</div>
    <div class="empty-state-heading">Ask about your document</div>
    <div class="empty-state-sub">Upload a PDF in the Sources pane, then ask a question below.</div>
    <div class="suggestion-chips">{suggestions_html}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close rag-center

    # ────────────────────────────────────────────────────────────────────────
    # RIGHT PANE — Evals / Configuration
    # ────────────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="rag-pane rag-pane-right" id="pane-right">
  <div class="pane-header">
    <span class="pane-header-icon">⚙️</span>
    <span class="pane-header-title">Evals &amp; Config</span>
    <button class="pane-collapse-btn" data-collapse="right" title="Collapse">›</button>
  </div>
  <div class="pane-body">
""", unsafe_allow_html=True)

    # ── Configuration sub-section ────────────────────────────────────────────
    with st.expander("⚙️ Configuration", expanded=True):
        index_name = st.text_input(
            "Index name",
            value=config.pinecone_index_name,
            help="Pinecone index to use",
            key="cfg_index",
        ).strip()
        top_k = st.number_input("Top K", min_value=1, max_value=10, value=config.top_k, step=1, key="cfg_topk")
        min_conf = st.slider(
            "Min confidence",
            min_value=0.0, max_value=1.0,
            value=float(config.min_confidence_score),
            step=0.01,
            key="cfg_conf",
        )
        config = config.with_overrides(
            pinecone_index_name=index_name or config.pinecone_index_name,
            top_k=int(top_k),
            min_confidence_score=float(min_conf),
        )
        if st.button("🗑 Clear conversation", key="clear_btn"):
            clear_conversation()
            st.rerun()

    # ── Retrieval Metrics sub-section ────────────────────────────────────────
    with st.expander("📊 Retrieval Metrics", expanded=True):
        metrics = st.session_state.latest_metrics
        if metrics:
            display = format_metrics_for_display(metrics)
            # Stat cards grid
            st.markdown(
                '<div class="stat-grid">'
                + _build_stat_card(display["Retrieved"], "Retrieved")
                + _build_stat_card(display["Relevant"], "Relevant")
                + _build_stat_card(display["Threshold"], "Threshold")
                + _build_stat_card(display["Average score (retrieval-score-based precision proxy)"], "Avg Score")
                + '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-label">Recall proxy</div>'
                f'<div style="font-size:22px;font-weight:700">{display["Recall proxy (retrieval-score-based)"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="section-label" style="margin-top:14px">Individual scores</div>', unsafe_allow_html=True)
            st.markdown(_build_scores_table(metrics.get("scores", [])), unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="font-size:13px;color:var(--text-muted)">No metrics yet — ask a question first.</p>',
                unsafe_allow_html=True,
            )

    st.markdown('</div></div>', unsafe_allow_html=True)  # close pane-body + pane-right

    # ── Close shell ──────────────────────────────────────────────────────────
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Source preview overlay ────────────────────────────────────────────────
    st.markdown("""
<div class="source-overlay-backdrop" id="src-overlay-backdrop">
  <div class="source-overlay">
    <div class="source-overlay-header">
      <span style="font-size:20px">📄</span>
      <span class="source-overlay-title" id="src-overlay-title"></span>
      <button class="source-overlay-close" id="src-overlay-close">✕</button>
    </div>
    <div class="source-overlay-body" id="src-overlay-body"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Inject JS ─────────────────────────────────────────────────────────────
    st.markdown(_COLLAPSE_JS, unsafe_allow_html=True)

    # ── Update source badge count via JS ──────────────────────────────────────
    src_count = 1 if st.session_state.document_name else 0
    st.markdown(
        f"<script>document.getElementById('src-badge') && "
        f"(document.getElementById('src-badge').textContent='{src_count}');</script>",
        unsafe_allow_html=True,
    )

    # ────────────────────────────────────────────────────────────────────────
    # Chat input + query handling
    # ────────────────────────────────────────────────────────────────────────
    source_ready = bool(st.session_state.namespace and st.session_state.source_active)
    placeholder = (
        "Ask a question about the uploaded PDF…"
        if source_ready
        else "Upload and index a PDF to start chatting…"
    )

    question = st.chat_input(placeholder, disabled=not source_ready, key="chat_input")

    if question is None:
        return

    question = question.strip()
    if not question:
        return

    if missing_keys:
        st.warning("Configure the required API keys before asking questions.")
        return

    try:
        embedding_service = build_embedding_service(config)
        llm_service = build_llm_service(config)
        vector_store = build_vector_store(config)

        with st.spinner("Retrieving…"):
            question_embedding = embedding_service.embed_query(question)
            retrieval_response = vector_store.query(
                embedding=question_embedding,
                namespace=st.session_state.namespace,
                top_k=config.top_k,
                min_confidence_score=config.min_confidence_score,
            )

        if retrieval_response["relevant_count"] == 0:
            answer = (
                _NO_INFO_PREFIX
                + " in the uploaded document to answer that question."
            )
        else:
            with st.spinner("Generating answer…"):
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
