# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then set AWS_BEARER_TOKEN_BEDROCK
```

For evaluation (optional):
```bash
pip install -r requirements-dev.txt
# ragas 0.2.15 has a broken import — apply this shim after install:
python -c "
import langchain_community, os, textwrap
shim = os.path.join(os.path.dirname(langchain_community.__file__), 'chat_models', 'vertexai.py')
open(shim, 'w').write('from langchain_google_vertexai import ChatVertexAI\n__all__ = [\"ChatVertexAI\"]\n')
print('shim written to', shim)
"
```

## Commands

```bash
# Ingest docs (one-time, ~30-60 min)
python main.py ingest

# Web UI
python main.py web --port 8080 --sources
python main.py web --port 8080 --sources --root-path https://<ngrok-url>  # behind tunnel

# Terminal chat
python main.py chat

# Resume failed ingestion (skips re-scraping)
python main.py embed

# DB status / reset
python main.py status
python main.py clear --force && python main.py ingest

# Lint / format
ruff check . --fix
ruff format .
```

## Architecture

The pipeline has two phases:

**Ingestion** (`python main.py ingest`):
1. `scraper.py` — BFS crawl of docs.pingidentity.com, yields `ScrapedPage` objects
2. `processor.py` — converts pages to full-text LangChain `Document` objects (no pre-chunking)
3. `vector_store.create_parent_retriever()` — two-level split: parent sections (4,000 chars) stored in `chroma_db/parent_store/` as UUID-named files; child chunks (512 chars) embedded via `BAAI/bge-large-en-v1.5` (CPU, 1024-dim) and stored in ChromaDB (HNSW index). Crash checkpoint: `chroma_db/pending_pages.json`.

**Query** (every user question):
1. `rag_chain._load_retriever()` — loads from module-level cache in `vector_store.py` (embeddings + retriever cached on first call, not reloaded per query)
2. ChromaDB HNSW search returns top `TOP_K * 2` child chunks
3. `ParentDocumentRetriever` deduplicates by `doc_id` → fetches full parent sections from disk
4. `rag_chain._rerank()` — cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescores all candidates jointly with the query, returns top `TOP_K` by relevance
5. `_format_docs()` assembles context (capped at `PARENT_CONTEXT_CHARS`, default 12,000)
6. Both paths use `query_stream()` → `ChatBedrockConverse.stream()` → token-by-token:
   - **Web UI**: Gradio `respond()` yields on each token
   - **Terminal**: `interactive_chat()` prints each token with `print(..., end="", flush=True)`

## Key Constraints

**AWS auth**: Uses `AWS_BEARER_TOKEN_BEDROCK`, not IAM or `ANTHROPIC_API_KEY`. `AWS_PROFILE=pingone` exists in the shell but breaks boto3 — `query_stream()` pops it before the boto3 call and restores in `finally`.

**LLM model**: Default `global.anthropic.claude-sonnet-4-6`. Override via `LLM_MODEL` env var.

**Gradio behind a tunnel**: Pass `--root-path <public-url>` or Gradio rejects requests (CSRF). The `root_path` param is wired through `web_ui.launch_web_ui()` → `demo.launch()`.

## Module-level caches (`vector_store.py`)

`_embeddings_cache`, `_retriever_cache`, `_vector_store_cache` — loaded once per process. The retriever holds ~54,000 child chunks; reloading it on every query adds ~10s. The web UI preloads the retriever synchronously in `launch_web_ui()` before the server starts. The cross-encoder reranker (`_reranker` in `rag_chain.py`) is also cached after first load.

## Evaluation

```bash
python main.py evaluate                   # hit rate only (fast)
python main.py evaluate --ragas           # + LLM-as-judge quality metrics (slow)
```

Test set format: `eval_testset.json` — array of `{question, expected_url, ground_truth}`. Results written to `eval_results.json`.

`gen-testset` reads from `chroma_db/parent_store/` (full parent sections, up to 4,000 chars each), not from ChromaDB child chunks — this gives the LLM enough content to generate substantive questions. Pages with fewer than 300 chars are skipped. During RAGAS evaluation, cases where the ground truth itself says "no content available" or similar are filtered out to avoid corrupting precision/recall scores.
