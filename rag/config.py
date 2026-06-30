"""Configuration settings for the Documentation Assistant."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Claude via Amazon Bedrock (uses local AWS credentials / IAM role)
LLM_MODEL = os.getenv("LLM_MODEL", "global.anthropic.claude-sonnet-4-6")

# Local embeddings (sentence-transformers, runs on CPU)
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

# Site branding — drives web UI title, system prompt, and terminal panel
DOCS_BASE_URL = os.getenv("DOCS_BASE_URL", "https://docs.example.com")
SITE_NAME = os.getenv("SITE_NAME", "")  # e.g. "PingIdentity" — derived from URL if blank


def _derive_site_name(url: str, override: str) -> str:
    if override:
        return override
    from urllib.parse import urlparse

    host = urlparse(url).hostname or url
    for prefix in ("www.", "docs."):
        if host.startswith(prefix):
            host = host[len(prefix) :]
            break
    return host.split(".")[0].capitalize()


SITE_NAME = _derive_site_name(DOCS_BASE_URL, SITE_NAME)

# ChromaDB settings
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")
MAX_PAGES = int(os.getenv("MAX_PAGES", "1500"))  # Limit pages to scrape
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.5"))  # Seconds between requests
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))  # Characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # Overlap to preserve context

# RAG settings
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "6"))  # Final chunks to send to LLM
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "18"))  # Candidates for MMR reranking
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))  # Max chars for context

# Page Index settings (better for URL-based citations)
# When enabled, uses parent-child retrieval: chunks for matching, full pages for context
USE_PAGE_INDEX = os.getenv("USE_PAGE_INDEX", "true").lower() in ("true", "1", "yes")
PAGE_CONTEXT_CHARS = int(os.getenv("PAGE_CONTEXT_CHARS", "3000"))  # Max chars per page in context

# Parent Document Retriever settings (recommended: better accuracy than flat chunking)
# How it works:
#   Ingest:  full page → split into parent sections (PARENT_CHUNK_SIZE chars)
#                      → each section split into child chunks (CHILD_CHUNK_SIZE chars)
#                      → child chunks embedded in Chroma (for fast semantic retrieval)
#                      → parent sections stored in LocalFileStore on disk
#   Query:   embed query → find top-k similar child chunks in Chroma
#                        → look up their parent section IDs
#                        → return full parent sections as LLM context
# Result: retrieval precision of small chunks + full section context for the LLM.
USE_PARENT_RETRIEVER = os.getenv("USE_PARENT_RETRIEVER", "true").lower() in ("true", "1", "yes")
PARENT_STORE_DIR = Path(os.getenv("PARENT_STORE_DIR", "./chroma_db/parent_store"))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "512"))  # Small: precise semantic matching
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "30"))
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "4000"))  # Large: full section context
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "200"))
PARENT_CONTEXT_CHARS = int(os.getenv("PARENT_CONTEXT_CHARS", "12000"))  # Max chars sent to LLM
