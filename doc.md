# PingIdentity Documentation Assistant - Setup Guide

Complete setup guide for new developers to run the PingIdentity Documentation Assistant locally.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [CLI Command Reference](#cli-command-reference)
6. [Docker Deployment](#docker-deployment)
7. [Tuning Guide](#tuning-guide)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.10+** installed
- **GitHub Personal Access Token** with the following permissions:
  - Read access to Copilot Chat
  - Read access to Copilot Editor Context
  - Read access to models
  - Read access to user copilot requests

### Getting a GitHub Token

1. Go to [GitHub Settings > Tokens](https://github.com/settings/tokens)
2. Click **Generate new token** → **Fine-grained personal access token**
3. Set expiration and grant the required permissions listed above
4. Copy the token (starts with `ghp_`)

---

## Installation

```bash
# 1. Clone and navigate to project
cd pingid-docs-assistant

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set GitHub token
export GITHUB_TOKEN='ghp_your_token_here'
```

**Optional**: Install dev dependencies for testing:
```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

After installation, run these commands to start using the application:

```bash
# Step 1: Ingest documentation (first time only, ~30-60 min)
python main.py ingest

# Step 2: Start the web UI
python main.py web
# Open http://localhost:7860 in your browser
```

That's it! You can now ask questions about PingIdentity documentation.

---

## Configuration

All settings are configured via environment variables. Create a `.env` file:

```bash
cp .env.example .env
```

### Required

| Variable | Description |
|----------|-------------|
| `GITHUB_TOKEN` | GitHub PAT with read access to Copilot Chat, Copilot Editor Context, models, and user copilot requests |

### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `openai/gpt-4o-mini` | LLM model for responses |
| `LLM_MIN_REQUEST_INTERVAL` | `1.0` | Seconds between LLM requests |
| `LLM_MAX_RETRIES` | `3` | Retry attempts on rate limit |

### Embedding Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LOCAL_EMBEDDINGS` | `true` | Use local embeddings (recommended) |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local model (384 dimensions) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | API model (if local disabled) |

### Vector Store Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector database location |
| `COLLECTION_NAME` | `pingid_docs` | ChromaDB collection name |

### Scraping Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCS_BASE_URL` | `https://docs.pingidentity.com` | Documentation site URL |
| `MAX_PAGES` | `1000` | Maximum pages to scrape |
| `REQUEST_DELAY` | `0.5` | Seconds between requests |
| `REQUEST_TIMEOUT` | `30` | Request timeout in seconds |

### Retrieval Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `6` | Final documents sent to LLM |
| `RETRIEVE_K` | `18` | Candidates for MMR reranking |
| `MAX_CONTEXT_CHARS` | `5000` | Max context chars (8K token limit) |
| `USE_PAGE_INDEX` | `true` | Group chunks by URL |
| `PAGE_CONTEXT_CHARS` | `3000` | Max chars per page |

---

## CLI Command Reference

### Data Ingestion Commands

| Command | Description |
|---------|-------------|
| `python main.py ingest` | Scrape and ingest documentation (default: 1000 pages) |
| `python main.py ingest --max-pages 500` | Limit pages to scrape |
| `python main.py ingest --force` | Clear database and re-ingest |
| `python main.py ingest --url <URL>` | Scrape custom documentation URL |
| `python main.py embed` | Resume embedding if ingestion failed |
| `python main.py embed --chunks-file <path>` | Embed from custom chunks file |

### Query Commands

| Command | Description |
|---------|-------------|
| `python main.py ask "your question"` | Single question (returns answer and exits) |
| `python main.py ask "question" --sources` | Include source URLs in response |
| `python main.py search "keyword"` | Search vectors directly (no LLM) |
| `python main.py search "keyword" --top-k 10` | Return more results |

### Interactive Commands

| Command | Description |
|---------|-------------|
| `python main.py chat` | Interactive terminal chat session |
| `python main.py chat --sources` | Show source URLs with responses |
| `python main.py web` | Start Gradio web UI at localhost:7860 |
| `python main.py web --port 8080` | Use custom port |
| `python main.py web --share` | Generate public URL (72 hours) |
| `python main.py web --sources` | Show source URLs in responses |
| `python main.py web --share --sources` | Public URL with sources |

**Chat session commands**: `sources` (toggle), `clear` (reset history), `quit`/`exit`

### Database Commands

| Command | Description |
|---------|-------------|
| `python main.py status` | Show database info (document count, etc.) |
| `python main.py clear` | Delete vector database (with confirmation) |
| `python main.py clear --force` | Delete without confirmation |

---

## Docker Deployment

### Build Image

```bash
docker build -t pingid-docs-assistant .

# For cloud deployment (Apple Silicon)
docker build --platform linux/amd64 -t pingid-docs-assistant .
```

### Run Container

```bash
# Ingest documentation first (if no existing database)
docker run -it --rm \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e GITHUB_TOKEN="$GITHUB_TOKEN" \
  pingid-docs-assistant \
  python main.py ingest

# Start web UI
docker run -d -p 7860:7860 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e GITHUB_TOKEN="$GITHUB_TOKEN" \
  --name pingid-assistant \
  pingid-docs-assistant \
  python main.py web --share --sources
```

### Docker Hub

```bash
docker tag pingid-docs-assistant your-username/pingid-docs-assistant:latest
docker push your-username/pingid-docs-assistant:latest
```

---

## Tuning Guide

### Get More Relevant Results
- Increase `TOP_K_RESULTS` (watch token limits)
- Set `RETRIEVE_K` to 2-3x of `TOP_K_RESULTS`

### Improve Chunk Context
- Increase `CHUNK_SIZE` (e.g., 1500)
- Set `CHUNK_OVERLAP` to ~20% of chunk size
- Re-ingest after changes: `python main.py ingest --force`

### Expand Documentation Coverage
- Increase `MAX_PAGES` in `.env`
- Or use: `python main.py ingest --max-pages 2000`

### Better Source Citations
- Keep `USE_PAGE_INDEX=true` (default)
- Groups related chunks by source URL

---

## Troubleshooting

### "No vector store found"
Run ingestion first:
```bash
python main.py ingest
```

### Rate limit errors (LLM)
The app auto-retries with backoff. If persistent:
- Wait a minute between questions
- Set `LLM_MIN_REQUEST_INTERVAL=3` in `.env`

### Rate limit errors (Embeddings)
Use local embeddings (default): `USE_LOCAL_EMBEDDINGS=true`

### Dimension mismatch error
Occurs when switching between local/API embeddings. Rebuild database:
```bash
python main.py clear --force
python main.py ingest
```

### Empty or irrelevant responses
1. Check database contents: `python main.py search "your query"`
2. Re-ingest with more pages: `python main.py ingest --max-pages 2000 --force`
3. Rephrase with specific product names (PingFederate, PingOne, etc.)

### Import errors
Ensure virtual environment is activated:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Gradio won't start
Check if port is in use:
```bash
lsof -i :7860
python main.py web --port 8080  # Use alternate port
```

---

## Tips

- **First run**: Initial ingestion takes 30-60 minutes for 1000 pages
- **Re-ingesting**: Use `--force` flag to clear and rebuild
- **Local embeddings**: Default `USE_LOCAL_EMBEDDINGS=true` avoids API rate limits
- **Share web UI**: Use `--share` for a public Gradio URL (valid 72 hours)
