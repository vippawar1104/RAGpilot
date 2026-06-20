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
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Newsreader:opsz,wght@6..72,500;6..72,700&display=swap');
    :root { --ink:#333333; --rust:#d4af37; --paper:#ffffff; --sage:#e8dcb8; }
    html, body, [class*="st-"] { font-family: "Newsreader", Georgia, serif; color: var(--ink); background-color: var(--paper); }
    code, .stMetricLabel { font-family: "DM Mono", monospace !important; }
    .block-container { max-width: 1480px; padding-top: 2rem; }
    h1 {
      letter-spacing: -0.04em;
      font-size: clamp(2.8rem, 6vw, 5.8rem) !important;
      line-height:.88;
    }
    h2 { letter-spacing: -0.025em; }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(212,175,55,.3); background-color: #faf8f5; }
    [data-testid="stChatMessage"] { border: 1px solid rgba(212,175,55,.3); border-radius: 4px; background-color: #ffffff; }
    .eyebrow {
      font-family:"DM Mono",monospace;
      text-transform:uppercase;
      letter-spacing:.12em;
      color:var(--rust);
    }
    .status-ready { color:#d4af37; }
    .source-card {
      border-left:3px solid var(--rust);
      padding:.65rem 1rem;
      margin:.45rem 0;
      background:rgba(212,175,55,.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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

with st.sidebar:
    st.markdown('<div class="eyebrow">RAGPilot / Local RAG</div>', unsafe_allow_html=True)
    st.header("Research console")
    ready_count = sum(doc.status == DocumentStatus.READY for doc in database.list_documents())
    st.metric("Indexed documents", ready_count)
    page = st.radio("Workspace", ["Ask", "Documents", "Retrieval lab", "Settings"])
    st.divider()
    st.caption("Local embeddings | Chroma HNSW | SQLite BM25 | cross-encoder reranking")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Evidence | {len(sources)} sources", expanded=False):
        for source in sources:
            location = f" | p. {source['page']}" if source["page"] else ""
            heading = f" | {source['heading']}" if source["heading"] else ""
            st.markdown(
                f"<div class='source-card'><strong>[{source['source']}] "
                f"{source['filename']}</strong>{location}{heading}<br>"
                f"<small>{source['excerpt']}</small></div>",
                unsafe_allow_html=True,
            )


if page == "Ask":
    st.title("Ask the archive.")
    st.caption("Hybrid retrieval finds candidates; a cross-encoder decides what reaches the model.")

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
                answer = "Add an LLM API key in Settings before asking a question."
                st.warning(answer)
                sources = []
            elif ready_count == 0:
                answer = (
                    "No indexed documents are available. Upload documents and wait for indexing."
                )
                st.warning(answer)
                sources = []
            else:
                try:
                    with st.status("Retrieving evidence...", expanded=False) as status:
                        trace = retrieval.retrieve(question)
                        context, included = retrieval.build_context(trace.final)
                        st.session_state.last_trace = trace
                        status.update(
                            label=(
                                f"Selected {len(included)} sources in "
                                f"{trace.timings_ms['total']:.0f} ms"
                            ),
                            state="complete",
                        )
                    generator = AnswerGenerator(
                        GenerationConfig(
                            api_key=st.session_state.api_key,
                            model=st.session_state.get("llm_model", settings.llm_model),
                            provider=st.session_state.get("llm_provider", settings.llm_provider),
                            base_url=st.session_state.get("llm_base_url", settings.llm_base_url),
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

elif page == "Documents":
    st.markdown('<div class="eyebrow">Corpus operations</div>', unsafe_allow_html=True)
    st.title("Build the archive.")
    st.caption("Upload individual files or a directory. Duplicate content is detected by SHA-256.")

    accept_multiple: bool | str = True
    try:
        from streamlit import __version__ as st_ver
        version_parts = [int(p) for p in st_ver.split(".") if p.isdigit()]
        if version_parts and (version_parts[0] > 1 or (version_parts[0] == 1 and len(version_parts) >= 2 and version_parts[1] >= 35)):
            accept_multiple = "directory"
    except Exception:
        pass

    uploaded_files = st.file_uploader(
        "Documents",
        accept_multiple_files=accept_multiple,
        type=[
            "pdf",
            "docx",
            "pptx",
            "xlsx",
            "html",
            "md",
            "txt",
            "csv",
            "json",
            "png",
            "jpg",
            "jpeg",
            "tiff",
        ],
    )
    if st.button("Queue files", type="primary", disabled=not uploaded_files):
        added = duplicates = 0
        progress = st.progress(0, text="Persisting uploads")
        for index, upload in enumerate(uploaded_files):
            _, duplicate = ingestion.enqueue_bytes(upload.name, upload.getvalue())
            duplicates += int(duplicate)
            added += int(not duplicate)
            progress.progress((index + 1) / len(uploaded_files), text=upload.name)
        progress.empty()
        st.success(f"Queued {added} files; skipped {duplicates} duplicates.")
        time.sleep(0.4)
        st.rerun()

    st.subheader("Document registry")
    documents = database.list_documents()
    if not documents:
        st.info("The archive is empty.")
    for document in documents:
        status_label = {
            "ready": "READY",
            "failed": "FAILED",
            "processing": "PROCESSING",
            "queued": "QUEUED",
        }.get(document.status, "UNKNOWN")
        with st.container(border=True):
            info, state, actions = st.columns([5, 2, 2])
            info.markdown(f"**{document.filename}**")
            info.caption(f"{document.size_bytes / 1024:.1f} KB | {document.chunk_count} chunks")
            state.markdown(f"**{status_label}**")
            if document.error:
                state.caption(document.error)
            with actions:
                if document.status == DocumentStatus.FAILED and st.button(
                    "Retry", key=f"retry-{document.id}", use_container_width=True
                ):
                    database.retry_document(document.id)
                    st.rerun()
                if st.button("Delete", key=f"delete-{document.id}", use_container_width=True):
                    ingestion.delete_document(document.id)
                    st.rerun()
    if any(doc.status in {DocumentStatus.QUEUED, DocumentStatus.PROCESSING} for doc in documents):
        st.caption("Indexing is active. Refresh to update status.")
        if st.button("Refresh status"):
            st.rerun()

elif page == "Retrieval lab":
    st.markdown('<div class="eyebrow">Pipeline observability</div>', unsafe_allow_html=True)
    st.title("Inspect every ranking decision.")
    debug_query = st.text_input("Run retrieval without calling the LLM")
    if st.button("Run retrieval", type="primary", disabled=not debug_query):
        try:
            st.session_state.last_trace = retrieval.retrieve(debug_query)
        except Exception as exc:
            st.error(str(exc))

    trace: RetrievalTrace | None = st.session_state.last_trace
    if trace:
        metric_columns = st.columns(len(trace.timings_ms))
        for column, (name, value) in zip(metric_columns, trace.timings_ms.items(), strict=False):
            column.metric(name.title(), f"{value:.0f} ms")
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
        st.info("Run a query here or ask a question to populate the retrieval trace.")

else:
    st.markdown('<div class="eyebrow">Runtime configuration</div>', unsafe_allow_html=True)
    st.title("Models and credentials.")
    st.warning("Credentials entered here remain in this Streamlit session and are not persisted.")
    st.session_state.api_key = st.text_input(
        "LLM API key", value=st.session_state.api_key, type="password"
    )
    provider_options = ["auto", "anthropic", "openai"]
    current_provider = st.session_state.get("llm_provider", settings.llm_provider)
    st.session_state.llm_provider = st.selectbox(
        "LLM provider",
        provider_options,
        index=provider_options.index(current_provider),
        help="Auto detects Anthropic keys by their sk-ant- prefix.",
    )
    st.session_state.llm_model = st.text_input(
        "LLM model", value=st.session_state.get("llm_model", settings.llm_model)
    )
    st.session_state.llm_base_url = (
        st.text_input(
            "Custom API base URL (optional)",
            value=st.session_state.get("llm_base_url", settings.llm_base_url) or "",
        )
        or None
    )
    st.divider()
    st.subheader("Local retrieval models")
    st.code(
        f"Embedding: {settings.embedding_model}\n"
        f"Reranker:  {settings.reranker_model}\n"
        f"Device:    {settings.device}"
    )
    st.caption("Change local retrieval models in .env, then restart the application and re-index.")
