# Documentation Assistant — Setup Guide

Generic RAG-based AI assistant for any documentation site. Point it at a URL, ingest, and query.

## Video


https://github.com/user-attachments/assets/9e2e3e08-44ea-4834-bd52-c75890adc05d


## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [CLI Command Reference](#cli-command-reference)
6. [Docker Deployment](#docker-deployment)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Tuning Guide](#tuning-guide)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.10+**
- **AWS account with Bedrock access** — Claude models must be [enabled in your AWS region](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
- **AWS credentials** — one of:
  - Standard AWS credentials in `~/.aws/credentials` (IAM user or assumed role) — works out of the box
  - `AWS_BEARER_TOKEN_BEDROCK` env var — short-lived bearer token for cross-account or CI use
- **`AWS_REGION`** set to a region where Bedrock is available (e.g. `us-east-1`)

---

## Installation

```bash
# 1. Navigate to project directory
cd docs-assistant

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Optional**: Install dev dependencies for testing:
```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

```bash
# Step 1: Set AWS credentials (if not already in ~/.aws/credentials)
export AWS_REGION=us-east-1
export AWS_BEARER_TOKEN_BEDROCK=<your-token>   # only if using bearer auth

# Step 2: Configure the target site
cp .env.example .env
# Edit .env: set DOCS_BASE_URL=https://docs.yoursite.com

# Step 3: Ingest documentation (first time only, ~30-60 min)
python main.py ingest

# Step 4: Start the web UI
python main.py web --port 8080
# Open http://localhost:8080 in your browser
```

---

## Configuration

All settings are configured via environment variables. Create a `.env` file:

```bash
cp .env.example .env
```

### Branding Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCS_BASE_URL` | `https://docs.example.com` | Documentation site to scrape |
| `SITE_NAME` | *(derived from URL)* | Display name in web UI and prompts (e.g. `Stripe`) |

### AWS / LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | *(required)* | AWS region where Bedrock is enabled (e.g. `us-east-1`) |
| `AWS_BEARER_TOKEN_BEDROCK` | *(optional)* | Short-lived bearer token; omit if using IAM credentials |
| `LLM_MODEL` | `global.anthropic.claude-sonnet-4-6` | Claude model via Amazon Bedrock |

### Embedding Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | sentence-transformers model (runs on CPU, 1024-dim) |

### Vector Store Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector database location |
| `COLLECTION_NAME` | `docs` | ChromaDB collection name |

### Scraping Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_PAGES` | `1500` | Maximum pages to scrape |
| `REQUEST_DELAY` | `0.5` | Seconds between requests |
| `REQUEST_TIMEOUT` | `30` | Request timeout in seconds |

### Retrieval Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `6` | Final documents sent to LLM |
| `RETRIEVE_K` | `18` | Candidates for MMR reranking |
| `MAX_CONTEXT_CHARS` | `5000` | Max context characters |
| `USE_PAGE_INDEX` | `true` | Group chunks by source URL |
| `PAGE_CONTEXT_CHARS` | `3000` | Max chars per page in context |

### Parent Document Retriever Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_PARENT_RETRIEVER` | `true` | Enable parent-child retrieval (recommended) |
| `PARENT_STORE_DIR` | `./chroma_db/parent_store` | Parent section storage |
| `CHILD_CHUNK_SIZE` | `512` | Child chunk size for semantic matching |
| `PARENT_CHUNK_SIZE` | `4000` | Parent section size for LLM context |
| `PARENT_CONTEXT_CHARS` | `12000` | Max parent context chars sent to LLM |

---

## CLI Command Reference

### Data Ingestion

| Command | Description |
|---------|-------------|
| `python main.py ingest` | Scrape and ingest documentation |
| `python main.py ingest --max-pages 500` | Limit pages to scrape |
| `python main.py ingest --force` | Clear database and re-ingest |
| `python main.py ingest --url <URL>` | Scrape a custom documentation URL |
| `python main.py embed` | Resume embedding if ingestion was interrupted |

### Interactive

| Command | Description |
|---------|-------------|
| `python main.py chat` | Interactive terminal chat session |
| `python main.py chat --sources` | Show source URLs with responses |
| `python main.py web` | Start Gradio web UI at localhost:7860 |
| `python main.py web --port 8080` | Use custom port |
| `python main.py web --share` | Generate public URL (72 hours) |
| `python main.py web --sources` | Show source URLs in web UI responses |
| `python main.py web --root-path <url>` | Set public base URL when behind a tunnel (e.g. ngrok) |

**Chat session commands**: `sources` (toggle), `clear` (reset history), `quit`/`exit`

### Evaluation

| Command | Description |
|---------|-------------|
| `python main.py gen-testset` | Generate eval test set from the indexed docs (auto-run after ingest) |
| `python main.py gen-testset --pages 30 --questions 3` | Sample more pages / questions |
| `python main.py gen-testset --output my_tests.json` | Write to a custom file |
| `python main.py evaluate` | Run retrieval hit rate benchmark (fast, no LLM) |
| `python main.py evaluate --ragas` | + RAGAS quality metrics (slow, needs `pip install ragas`) |

`gen-testset` reads full parent sections from `chroma_db/parent_store/` (up to 4,000 chars per section) and uses the LLM to generate realistic questions, writing `eval_testset.json` with `expected_url` pre-filled so `evaluate` works immediately. It runs automatically at the end of `ingest`, but can be re-run any time against an existing index. Pages with fewer than 300 chars of content are skipped to avoid generating unanswerable questions.

### Database

| Command | Description |
|---------|-------------|
| `python main.py status` | Show database info and document count |
| `python main.py clear` | Delete vector database (with confirmation) |
| `python main.py clear --force` | Delete without confirmation |

---

## Docker Deployment

### Build Image

```bash
docker build -t docs-assistant .

# For cloud deployment from Apple Silicon
docker build --platform linux/amd64 -t docs-assistant .
```

### Run Container

```bash
# Ingest documentation
docker run -it --rm \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e DOCS_BASE_URL="https://docs.yoursite.com" \
  -e AWS_BEARER_TOKEN_BEDROCK="$AWS_BEARER_TOKEN_BEDROCK" \
  -e AWS_REGION="$AWS_REGION" \
  docs-assistant \
  python main.py ingest

# Start web UI
docker run -d -p 8080:8080 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e DOCS_BASE_URL="https://docs.yoursite.com" \
  -e AWS_BEARER_TOKEN_BEDROCK="$AWS_BEARER_TOKEN_BEDROCK" \
  -e AWS_REGION="$AWS_REGION" \
  --name docs-assistant \
  docs-assistant \
  python main.py web --port 8080
```

---

## Evaluation Metrics

The evaluation pipeline runs two independent benchmarks. Results are written to `eval_results.json`.

### Retrieval Benchmark: Hit Rate and MRR

These metrics measure whether the retriever fetches the right document, **without involving the LLM**. They are fast to run and don't require ground-truth answers — only `expected_url` in the test set.

**Hit Rate @k** — Did the correct page appear anywhere in the top-k results?

Each question scores 1 if the expected URL appears in any of the top-k retrieved documents, 0 otherwise. Hit Rate is the fraction of questions that scored 1. It tells you: *does the retriever ever find the right page?*

**MRR (Mean Reciprocal Rank)** — How high up in the list was the correct page?

Each question scores `1 / rank` where rank is the position of the first correct hit. MRR is the average across all questions. A question where the correct doc is at rank 1 scores 1.0; at rank 3 it scores 0.33. MRR tells you: *when the retriever finds it, is it near the top?*

A high hit rate with low MRR means the right doc is consistently buried at rank 5–6. That's bad for generation quality because the LLM context fills up with noise before reaching the relevant section.

| Target | Hit Rate @6 | MRR |
|--------|-------------|-----|
| Good | > 0.80 | > 0.60 |

### RAGAS Benchmark: Generation Quality

RAGAS uses an LLM-as-judge to evaluate the full RAG pipeline. It requires both `expected_url` and `ground_truth` answers in the test set, and calls the LLM for each evaluation question — expect it to take several minutes.

Install RAGAS first:
```bash
pip install -r requirements-dev.txt
```

| Metric | What it measures | Target |
|--------|-----------------|--------|
| `context_precision` | Of the retrieved chunks sent to the LLM, what fraction were actually relevant to the question? Noise chunks hurt this score. | > 0.80 |
| `context_recall` | Did the retriever fetch all the information needed to answer the question? Missing a key doc hurts this score. | > 0.75 |
| `faithfulness` | Does the generated answer stay within the retrieved context, or does the LLM hallucinate facts not present in the docs? | > 0.85 |
| `answer_relevancy` | Does the answer actually address what was asked, or does it hedge, deflect, or answer a different question? | > 0.80 |

**Interpreting RAGAS scores**: Use them as a relative comparator and regression detector, not an absolute quality certificate. The judge LLM evaluates internal consistency between context and answer — it cannot verify factual accuracy against the real world. A consistently high score across multiple test runs with diverse questions indicates a well-functioning pipeline, but complement it with periodic manual testing using real user questions.

---

## Tuning Guide

### Get More Relevant Results
- Increase `TOP_K_RESULTS` (e.g. `8`)
- Set `RETRIEVE_K` to 2–3× `TOP_K_RESULTS`

### Improve Chunk Context
- Increase `PARENT_CHUNK_SIZE` (e.g. `6000`) for more context per retrieved section
- Re-ingest after changes: `python main.py ingest --force`

### Expand Documentation Coverage
- Increase `MAX_PAGES` in `.env` or pass `--max-pages 2000`

---

## Troubleshooting

### "No vector store found"
```bash
python main.py ingest
```

### Dimension mismatch error
Occurs when switching embedding models. Rebuild the database:
```bash
python main.py clear --force && python main.py ingest
```

### Empty or irrelevant responses
1. Re-ingest with more pages: `python main.py ingest --max-pages 2000 --force`
2. Use specific terms from the target documentation in queries

### Port already in use
```bash
lsof -ti :8080 | xargs kill -9
python main.py web --port 8080
```

### Import errors
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### `evaluate --ragas` crashes with `ImportError: cannot import name 'ChatVertexAI'`

ragas 0.2.x has a broken import in `langchain_community`. Apply this one-time shim after install:

```bash
python -c "
import langchain_community, os
shim = os.path.join(os.path.dirname(langchain_community.__file__), 'chat_models', 'vertexai.py')
open(shim, 'w').write('from langchain_google_vertexai import ChatVertexAI\n__all__ = [\"ChatVertexAI\"]\n')
print('shim written to', shim)
"
```
