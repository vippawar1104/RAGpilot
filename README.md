# RAGPilot

A local, evidence-first RAG application with bulk document ingestion, structure-aware chunks,
Chroma vector search, SQLite FTS5/BM25 retrieval, reciprocal-rank fusion, cross-encoder reranking,
parent context expansion, and grounded LLM answers with citations.

## Capabilities

- Upload multiple files or an entire directory from Streamlit.
- Parse PDF, Office, HTML, Markdown, text, tables, and images through Docling.
- Deduplicate documents with SHA-256 and process them using a persistent background queue.
- Keep embeddings, reranking, Chroma, originals, metadata, and keyword search local.
- Combine semantic and lexical candidates with Reciprocal Rank Fusion.
- Rerank candidates with a local cross-encoder before building context.
- Inspect every retrieval stage without spending LLM tokens.
- Use Anthropic, OpenAI, or an OpenAI-compatible endpoint; API keys remain session-only.

## Setup

Python 3.11 or 3.12 is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
streamlit run app.py
```

*Alternatively, if running via system or Anaconda Python, run with `PYTHONPATH` set to `src`:*
```bash
PYTHONPATH=src streamlit run app.py
```

The embedding and reranker models download on first use. On Apple Silicon, set `RAG_DEVICE=mps`
if the installed PyTorch build supports it. Keep `cpu` when compatibility matters more than speed.

## Background worker

The Streamlit app starts one in-process daemon worker. For ingestion without the UI, run:

```bash
advanced-rag-worker
```

SQLite job claiming is atomic, so multiple worker processes will not claim the same queued job.

## Configuration

Copy `.env.example` to `.env`. Important settings:

| Variable | Purpose |
|---|---|
| `RAG_LLM_API_KEY` | Optional server-side default key |
| `RAG_LLM_PROVIDER` | `auto`, `anthropic`, or `openai` |
| `RAG_LLM_MODEL` | Chat-completions model name |
| `RAG_LLM_BASE_URL` | Optional OpenAI-compatible endpoint |
| `RAG_EMBEDDING_MODEL` | Sentence Transformers embedding model |
| `RAG_RERANKER_MODEL` | Sentence Transformers CrossEncoder model |
| `RAG_DEVICE` | `cpu`, `mps`, or `cuda` |

Changing the embedding model requires deleting `data/chroma`, `data/app.db`, and re-indexing the
corpus because vector dimensions and semantics are model-specific.

## Retrieval flow

```text
query -> dense top 40 + BM25 top 40 -> RRF -> top 30 cross-encoder
      -> source diversity -> parent expansion -> context budget -> grounded answer
```

## Tests

```bash
pytest
ruff check .
```

## Current scope

This release implements the high-value retrieval core. A graph retrieval adapter can be added
without changing ingestion or generation, but graph extraction is intentionally not enabled by
default because it adds substantial indexing cost and should be justified by multi-hop evaluation.
