"""Vector store management using ChromaDB."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from tqdm import tqdm

# Try new package first, fall back to deprecated one
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from . import config

if TYPE_CHECKING:
    from langchain_core.documents import Document

console = Console()

# Rate limiting for GitHub Models API (not needed for local embeddings)
EMBEDDING_DELAY = float(os.getenv("EMBEDDING_DELAY", "0.0"))  # No delay for local


def get_embeddings(use_local: bool | None = None) -> Embeddings:
    """Get embeddings model - local or API based on config.

    Args:
        use_local: Override config setting. None uses config.USE_LOCAL_EMBEDDINGS.

    Returns:
        Embeddings instance (HuggingFace local or OpenAI API)
    """
    if use_local is None:
        use_local = config.USE_LOCAL_EMBEDDINGS

    if use_local:
        console.print(f"[dim]Using local embeddings: {config.LOCAL_EMBEDDING_MODEL}[/dim]")
        return HuggingFaceEmbeddings(
            model_name=config.LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},  # Use "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        if not config.GITHUB_TOKEN:
            raise ValueError(
                "GitHub token required for API embeddings. "
                "Set GITHUB_TOKEN or use USE_LOCAL_EMBEDDINGS=true"
            )
        console.print(f"[dim]Using API embeddings: {config.EMBEDDING_MODEL}[/dim]")
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.GITHUB_TOKEN,
            openai_api_base=config.GITHUB_MODELS_URL,
        )


def create_vector_store(
    documents: list[Document],
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
    batch_size: int | None = None,
) -> Chroma:
    """Create a new ChromaDB vector store from documents.

    Args:
        documents: List of LangChain Document objects to embed
        collection_name: Name for the ChromaDB collection
        persist_directory: Directory to persist the database
        batch_size: Number of documents to embed at once (auto-set based on local/API)

    Returns:
        Chroma vector store instance
    """
    persist_dir = Path(persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()

    # Set batch size and delay based on embedding type
    use_local = config.USE_LOCAL_EMBEDDINGS
    if batch_size is None:
        batch_size = 500 if use_local else 50  # Larger batches for local
    delay = 0.0 if use_local else EMBEDDING_DELAY

    console.print(
        f"[bold blue]Creating vector store with {len(documents)} documents...[/bold blue]"
    )
    console.print(f"[dim]Collection: {collection_name}, Persist: {persist_dir}[/dim]")
    if not use_local:
        console.print(f"[dim]Batch size: {batch_size}, Delay: {delay}s between batches[/dim]")

    # Process in batches to show progress and handle rate limits
    if len(documents) <= batch_size:
        # Small dataset - create directly
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(persist_dir),
        )
    else:
        # Large dataset - batch process with rate limiting
        console.print(f"[dim]Processing in batches of {batch_size}...[/dim]")

        # Create initial store with first batch
        first_batch = documents[:batch_size]
        vector_store = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(persist_dir),
        )

        # Add remaining batches with rate limiting
        remaining = documents[batch_size:]
        total_batches = (len(remaining) + batch_size - 1) // batch_size

        for i in tqdm(
            range(0, len(remaining), batch_size),
            desc="Embedding batches",
            unit="batch",
            total=total_batches,
        ):
            # Rate limit delay (only for API embeddings)
            if delay > 0:
                time.sleep(delay)

            batch = remaining[i : i + batch_size]

            # Retry logic for rate limits (API embeddings only)
            if use_local:
                vector_store.add_documents(batch)
            else:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        vector_store.add_documents(batch)
                        break
                    except Exception as e:
                        if "Too many requests" in str(e) or "429" in str(e):
                            wait_time = max(delay, 1.0) * (2**attempt)  # Exponential backoff
                            console.print(f"[yellow]Rate limited, waiting {wait_time}s...[/yellow]")
                            time.sleep(wait_time)
                            if attempt == max_retries - 1:
                                raise
                        else:
                            raise

    console.print(f"[bold green]Vector store created with {len(documents)} documents[/bold green]")

    return vector_store


def load_vector_store(
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
) -> Chroma:
    """Load an existing ChromaDB vector store.

    Args:
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory where the database is persisted

    Returns:
        Chroma vector store instance
    """
    persist_dir = Path(persist_directory)

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. "
            "Run 'ingest' command first to create the database."
        )

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    # Get count of documents
    collection = vector_store._collection
    count = collection.count()
    console.print(f"[bold green]Loaded vector store with {count} documents[/bold green]")

    return vector_store


def delete_vector_store(
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
) -> bool:
    """Delete the vector store directory.

    Args:
        persist_directory: Directory where the database is persisted

    Returns:
        True if deleted, False if not found
    """
    import shutil

    persist_dir = Path(persist_directory)

    if persist_dir.exists():
        shutil.rmtree(persist_dir)
        console.print(f"[bold yellow]Deleted vector store at {persist_dir}[/bold yellow]")
        return True

    console.print(f"[dim]No vector store found at {persist_dir}[/dim]")
    return False


def similarity_search(
    query: str,
    k: int = config.TOP_K_RESULTS,
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
) -> list[Document]:
    """Perform similarity search on the vector store.

    Args:
        query: Search query text
        k: Number of results to return
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory where the database is persisted

    Returns:
        List of most similar documents
    """
    vector_store = load_vector_store(collection_name, persist_directory)
    return vector_store.similarity_search(query, k=k)
