from __future__ import annotations

import threading
import time

import streamlit as st

from advanced_rag.config import get_settings
from advanced_rag.database import Database
from advanced_rag.generation import AnswerGenerator, GenerationConfig, source_summary
from advanced_rag.ingestion import IngestionService
from advanced_rag.models import DocumentStatus, RetrievalTrace
from advanced_rag.retrieval import RetrievalEngine

st.set_page_config(
    page_title="RAGPilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium dark-mode CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Design tokens ─────────────────────────────────────────────── */
    :root {
        --bg-primary:    #0E1117;
        --bg-secondary:  #161B22;
        --bg-card:       rgba(30, 37, 48, 0.65);
        --text-primary:  #E6EDF3;
        --text-secondary:#8B949E;
        --accent:        #6C63FF;
        --accent-glow:   rgba(108,99,255,0.25);
        --accent-light:  #A5A0FF;
        --success:       #3FB950;
        --warning:       #D29922;
        --danger:        #F85149;
        --border:        rgba(139,148,158,0.15);
        --radius:        10px;
    }

    /* ── Global typography ─────────────────────────────────────────── */
    html, body, [class*="st-"] {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }
    code, .stCodeBlock, pre, .stMetricLabel {
        font-family: "JetBrains Mono", "Fira Code", monospace !important;
    }

    /* ── Layout ────────────────────────────────────────────────────── */
    .block-container {
        max-width: 1420px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12161D 0%, #0E1117 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
    }

    /* ── Headings ──────────────────────────────────────────────────── */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        font-size: clamp(2.2rem, 5vw, 3.6rem) !important;
        line-height: 1.1 !important;
        background: linear-gradient(135deg, #E6EDF3 0%, #A5A0FF 50%, #6C63FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem !important;
    }
    h2 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: var(--text-primary) !important;
    }
    h3 {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    /* ── Eyebrow label ─────────────────────────────────────────────── */
    .eyebrow {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--accent-light);
        margin-bottom: 0.3rem;
        padding: 4px 10px;
        background: var(--accent-glow);
        border-radius: 6px;
        display: inline-block;
    }

    /* ── Sidebar brand ─────────────────────────────────────────────── */
    .sidebar-brand {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--accent-light) !important;
        padding: 6px 12px;
        background: var(--accent-glow);
        border-radius: 8px;
        border: 1px solid rgba(108,99,255,0.2);
        display: inline-block;
        margin-bottom: 0.6rem;
    }

    /* ── Chat messages ─────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        backdrop-filter: blur(12px);
        padding: 1rem !important;
    }

    /* ── Source cards ───────────────────────────────────────────────── */
    .source-card {
        border-left: 3px solid var(--accent);
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        background: var(--bg-card);
        border-radius: 0 var(--radius) var(--radius) 0;
        backdrop-filter: blur(8px);
        transition: border-left-color 0.2s ease, background 0.2s ease;
        color: var(--text-primary) !important;
    }
    .source-card:hover {
        border-left-color: var(--accent-light);
        background: rgba(108,99,255,0.08);
    }
    .source-card strong {
        color: var(--accent-light) !important;
    }
    .source-card small {
        color: var(--text-secondary) !important;
    }

    /* ── Status badges ─────────────────────────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-ready   { background: rgba(63,185,80,0.15); color: #3FB950; border: 1px solid rgba(63,185,80,0.3); }
    .badge-queued   { background: rgba(210,153,34,0.15); color: #D29922; border: 1px solid rgba(210,153,34,0.3); }
    .badge-processing { background: rgba(108,99,255,0.15); color: #A5A0FF; border: 1px solid rgba(108,99,255,0.3); }
    .badge-failed   { background: rgba(248,81,73,0.15); color: #F85149; border: 1px solid rgba(248,81,73,0.3); }

    /* ── Metric cards ──────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 0.8rem 1rem !important;
        backdrop-filter: blur(8px);
    }
    [data-testid="stMetricValue"] {
        color: var(--accent-light) !important;
        font-weight: 700 !important;
    }

    /* ── Buttons ────────────────────────────────────────────────────── */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #6C63FF, #8B83FF) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 8px var(--accent-glow) !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px var(--accent-glow) !important;
    }
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"] {
        background: transparent !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="stBaseButton-secondary"]:hover {
        border-color: var(--accent) !important;
        color: var(--accent-light) !important;
    }

    /* ── Input fields ──────────────────────────────────────────────── */
    .stTextInput input, .stSelectbox select, [data-baseweb="select"],
    .stTextInput > div > div > input {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus,
    .stTextInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-glow) !important;
    }
    .stTextInput label, .stSelectbox label, .stFileUploader label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }

    /* ── Tabs ──────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-secondary);
        border-radius: var(--radius);
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-glow) !important;
        color: var(--accent-light) !important;
    }

    /* ── Containers / Expanders ─────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]{
        color: var(--text-primary) !important;
    }

    /* ── Dataframe ─────────────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }

    /* ── Divider ───────────────────────────────────────────────────── */
    hr {
        border-color: var(--border) !important;
    }

    /* ── Warning / Info / Success boxes ─────────────────────────────── */
    .stAlert {
        border-radius: var(--radius) !important;
    }

    /* ── Progress bar ──────────────────────────────────────────────── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6C63FF, #A5A0FF) !important;
    }

    /* ── Chat input ────────────────────────────────────────────────── */
    [data-testid="stChatInput"] textarea {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-glow) !important;
    }

    /* ── File uploader ─────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: var(--bg-card) !important;
        border: 1px dashed var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
    }

    /* ── Sidebar metric override ───────────────────────────────────── */
    section[data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(108,99,255,0.08) !important;
        border: 1px solid rgba(108,99,255,0.15) !important;
    }

    /* ── Scrollbar ─────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

    /* ── Container borders ─────────────────────────────────────────── */
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"]) {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }

    /* ── Caption text ──────────────────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-secondary) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached services
# ---------------------------------------------------------------------------
@st.cache_resource
def services():
    settings = get_settings()
    database = Database(settings.database_path)
    ingestion = IngestionService(settings, database)
    retrieval = RetrievalEngine(settings, database)
    return settings, database, ingestion, retrieval


@st.cache_resource
def start_worker() -> threading.Thread:
    settings = get_settings()

    def consume() -> None:
        database = Database(settings.database_path)
        worker = IngestionService(settings, database)
        while True:
            if worker.process_next() is None:
                time.sleep(1.5)

    thread = threading.Thread(target=consume, name="rag-ingestion-worker", daemon=True)
    thread.start()
    return thread


settings, database, ingestion, retrieval = services()
start_worker()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = settings.llm_api_key
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">🚀 RAGPilot</div>',
        unsafe_allow_html=True,
    )
    st.header("Research console")
    ready_count = sum(doc.status == DocumentStatus.READY for doc in database.list_documents())
    st.metric("Indexed documents", ready_count)
    page = st.radio("Workspace", ["Ask", "Documents", "Retrieval lab", "Settings"])
    st.divider()
    st.caption("Local embeddings · Chroma HNSW · SQLite BM25 · Cross-encoder reranking")


# ---------------------------------------------------------------------------
# Helper: source cards
# ---------------------------------------------------------------------------
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Evidence — {len(sources)} sources", expanded=False):
        for source in sources:
            location = f" · p. {source['page']}" if source["page"] else ""
            heading = f" · {source['heading']}" if source["heading"] else ""
            st.markdown(
                f"<div class='source-card'><strong>[{source['source']}] "
                f"{source['filename']}</strong>{location}{heading}<br>"
                f"<small>{source['excerpt']}</small></div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Page: Ask
# ---------------------------------------------------------------------------
if page == "Ask":
    st.markdown('<div class="eyebrow">Evidence-first answers</div>', unsafe_allow_html=True)
    st.title("Ask the archive.")
    st.caption(
        "Hybrid retrieval finds candidates · a cross-encoder decides what reaches the model."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    if question := st.chat_input("Ask a question grounded in your documents"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            if not st.session_state.api_key:
                answer = "⚠️ Add an LLM API key in **Settings** before asking a question."
                st.warning(answer)
                sources = []
            elif ready_count == 0:
                answer = (
                    "📂 No indexed documents are available. "
                    "Upload documents in the **Documents** tab and wait for indexing."
                )
                st.warning(answer)
                sources = []
            else:
                try:
                    with st.status("🔍 Retrieving evidence...", expanded=False) as status:
                        trace = retrieval.retrieve(question)
                        context, included = retrieval.build_context(trace.final)
                        st.session_state.last_trace = trace
                        status.update(
                            label=(
                                f"✅ Selected {len(included)} sources in "
                                f"{trace.timings_ms['total']:.0f} ms"
                            ),
                            state="complete",
                        )
                    generator = AnswerGenerator(
                        GenerationConfig(
                            api_key=st.session_state.api_key,
                            model=st.session_state.get("llm_model", settings.llm_model),
                            provider=st.session_state.get(
                                "llm_provider", settings.llm_provider
                            ),
                            base_url=st.session_state.get(
                                "llm_base_url", settings.llm_base_url
                            ),
                        )
                    )
                    history = [
                        {"role": item["role"], "content": item["content"]}
                        for item in st.session_state.messages[:-1]
                    ]
                    answer = st.write_stream(generator.stream(question, context, history))
                    sources = source_summary(included)
                    render_sources(sources)
                except Exception as exc:
                    answer = f"Request failed: {exc}"
                    sources = []
                    st.error(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

# ---------------------------------------------------------------------------
# Page: Documents
# ---------------------------------------------------------------------------
elif page == "Documents":
    st.markdown('<div class="eyebrow">Corpus operations</div>', unsafe_allow_html=True)
    st.title("Build the archive.")
    st.caption("Upload individual files or drag a directory. Duplicates detected by SHA-256.")

    accept_multiple: bool | str = True
    try:
        from streamlit import __version__ as st_ver

        version_parts = [int(p) for p in st_ver.split(".") if p.isdigit()]
        if version_parts and (
            version_parts[0] > 1
            or (version_parts[0] == 1 and len(version_parts) >= 2 and version_parts[1] >= 35)
        ):
            accept_multiple = "directory"
    except Exception:
        pass

    uploaded_files = st.file_uploader(
        "Documents",
        accept_multiple_files=accept_multiple,
        type=[
            "pdf", "docx", "pptx", "xlsx", "html",
            "md", "txt", "csv", "json",
            "png", "jpg", "jpeg", "tiff",
        ],
    )
    if st.button("⬆️ Queue files", type="primary", disabled=not uploaded_files):
        added = duplicates = 0
        progress = st.progress(0, text="Persisting uploads…")
        for index, upload in enumerate(uploaded_files):
            _, duplicate = ingestion.enqueue_bytes(upload.name, upload.getvalue())
            duplicates += int(duplicate)
            added += int(not duplicate)
            progress.progress((index + 1) / len(uploaded_files), text=upload.name)
        progress.empty()
        st.success(f"Queued **{added}** files · skipped **{duplicates}** duplicates.")
        time.sleep(0.4)
        st.rerun()

    st.subheader("Document registry")
    documents = database.list_documents()
    if not documents:
        st.info("The archive is empty. Upload documents above to get started.")
    for document in documents:
        badge_class = {
            "ready": "badge-ready",
            "failed": "badge-failed",
            "processing": "badge-processing",
            "queued": "badge-queued",
        }.get(document.status, "badge-queued")
        status_label = document.status.upper() if isinstance(document.status, str) else str(document.status).split(".")[-1].upper()
        with st.container(border=True):
            info, state, actions = st.columns([5, 2, 2])
            info.markdown(f"**📄 {document.filename}**")
            info.caption(f"{document.size_bytes / 1024:.1f} KB · {document.chunk_count} chunks")
            state.markdown(
                f"<span class='status-badge {badge_class}'>{status_label}</span>",
                unsafe_allow_html=True,
            )
            if document.error:
                state.caption(f"❌ {document.error}")
            with actions:
                if document.status == DocumentStatus.FAILED and st.button(
                    "🔄 Retry", key=f"retry-{document.id}", use_container_width=True
                ):
                    database.retry_document(document.id)
                    st.rerun()
                if st.button(
                    "🗑️ Delete", key=f"delete-{document.id}", use_container_width=True
                ):
                    ingestion.delete_document(document.id)
                    st.rerun()
    if any(
        doc.status in {DocumentStatus.QUEUED, DocumentStatus.PROCESSING} for doc in documents
    ):
        st.caption("⏳ Indexing is active. Refresh to update status.")
        if st.button("🔄 Refresh status"):
            st.rerun()

# ---------------------------------------------------------------------------
# Page: Retrieval lab
# ---------------------------------------------------------------------------
elif page == "Retrieval lab":
    st.markdown(
        '<div class="eyebrow">Pipeline observability</div>', unsafe_allow_html=True
    )
    st.title("Inspect every ranking decision.")
    st.caption("Run retrieval without calling the LLM to debug your pipeline.")
    debug_query = st.text_input("Query", placeholder="Enter a search query…")
    if st.button("🔍 Run retrieval", type="primary", disabled=not debug_query):
        try:
            st.session_state.last_trace = retrieval.retrieve(debug_query)
        except Exception as exc:
            st.error(str(exc))

    trace: RetrievalTrace | None = st.session_state.last_trace
    if trace:
        metric_columns = st.columns(len(trace.timings_ms))
        for column, (name, value) in zip(
            metric_columns, trace.timings_ms.items(), strict=False
        ):
            column.metric(name.replace("_", " ").title(), f"{value:.0f} ms")
        tabs = st.tabs(["Final", "Fused", "Dense", "BM25"])
        datasets = [trace.final, trace.fused, trace.dense, trace.lexical]
        for tab, results in zip(tabs, datasets, strict=False):
            with tab:
                st.dataframe(
                    [
                        {
                            "file": item.filename,
                            "page": item.page,
                            "heading": item.heading,
                            "dense rank": item.dense_rank,
                            "BM25 rank": item.lexical_rank,
                            "fusion": round(item.score, 5),
                            "reranker": item.reranker_score,
                            "text": item.text[:350],
                        }
                        for item in results
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("Run a query above or ask a question on the **Ask** page to populate the trace.")

# ---------------------------------------------------------------------------
# Page: Settings
# ---------------------------------------------------------------------------
else:
    st.markdown(
        '<div class="eyebrow">Runtime configuration</div>', unsafe_allow_html=True
    )
    st.title("Models and credentials.")
    st.warning(
        "🔒 Credentials entered here remain in this Streamlit session and are **not** persisted."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.api_key = st.text_input(
            "🔑 LLM API key", value=st.session_state.api_key, type="password"
        )
        provider_options = ["auto", "anthropic", "openai"]
        current_provider = st.session_state.get("llm_provider", settings.llm_provider)
        st.session_state.llm_provider = st.selectbox(
            "🏢 LLM provider",
            provider_options,
            index=provider_options.index(current_provider),
            help="Auto detects Anthropic keys by their `sk-ant-` prefix.",
        )
    with col2:
        st.session_state.llm_model = st.text_input(
            "🤖 LLM model",
            value=st.session_state.get("llm_model", settings.llm_model),
        )
        st.session_state.llm_base_url = (
            st.text_input(
                "🌐 Custom API base URL (optional)",
                value=st.session_state.get("llm_base_url", settings.llm_base_url) or "",
            )
            or None
        )

    st.divider()
    st.subheader("⚙️ Local retrieval models")
    st.code(
        f"Embedding:  {settings.embedding_model}\n"
        f"Reranker:   {settings.reranker_model}\n"
        f"Device:     {settings.device}"
    )
    st.caption(
        "Change local retrieval models in `.env`, then restart the application and re-index."
    )
