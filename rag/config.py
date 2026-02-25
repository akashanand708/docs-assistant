"""Configuration settings for the PingIdentity Documentation Assistant."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GitHub Models API settings
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_MODELS_URL = "https://models.github.ai/inference"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")  # Faster, higher rate limits
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Local embedding settings (no API rate limits)
# Set USE_LOCAL_EMBEDDINGS=true to use local sentence-transformers model
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() in ("true", "1", "yes")
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)  # Fast & good quality

# ChromaDB settings
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pingid_docs")

# Scraping settings
DOCS_BASE_URL = os.getenv("DOCS_BASE_URL", "https://docs.pingidentity.com")
MAX_PAGES = int(os.getenv("MAX_PAGES", "1500"))  # Limit pages to scrape
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.5"))  # Seconds between requests
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# Document processing settings
# Smaller chunks to fit GitHub Models 8000 token limit
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))  # Characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # Overlap to preserve context

# RAG settings (tuned for GitHub Models 8000 token limit)
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "6"))  # Final chunks to send to LLM
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "18"))  # Candidates for MMR reranking
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))  # Max chars for context

# Page Index settings (better for URL-based citations)
# When enabled, uses parent-child retrieval: chunks for matching, full pages for context
USE_PAGE_INDEX = os.getenv("USE_PAGE_INDEX", "true").lower() in ("true", "1", "yes")
PAGE_CONTEXT_CHARS = int(os.getenv("PAGE_CONTEXT_CHARS", "3000"))  # Max chars per page in context

# LLM rate limiting (gpt-4o-mini has higher limits than gpt-4.1)
LLM_MIN_REQUEST_INTERVAL = float(
    os.getenv("LLM_MIN_REQUEST_INTERVAL", "1.0")
)  # Seconds between requests
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))  # Retry on rate limit
