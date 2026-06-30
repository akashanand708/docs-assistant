"""Vector store management using ChromaDB."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from tqdm import tqdm

# Try new package first, fall back to deprecated one
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from .. import config

if TYPE_CHECKING:
    from langchain_core.documents import Document

console = Console()

# Module-level caches — loaded once per process, reused on every query
_embeddings_cache: Embeddings | None = None
_retriever_cache: dict[tuple, ParentDocumentRetriever] = {}
_vector_store_cache: dict[tuple, Chroma] = {}


def get_embeddings() -> Embeddings:
    """Get local HuggingFace embeddings model (cached after first load)."""
    global _embeddings_cache
    if _embeddings_cache is None:
        console.print(f"[dim]Loading embeddings model: {config.LOCAL_EMBEDDING_MODEL}[/dim]")
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=config.LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_cache


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

    if batch_size is None:
        batch_size = 500

    console.print(
        f"[bold blue]Creating vector store with {len(documents)} documents...[/bold blue]"
    )
    console.print(f"[dim]Collection: {collection_name}, Persist: {persist_dir}[/dim]")

    # Process in batches to show progress
    if len(documents) <= batch_size:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(persist_dir),
        )
    else:
        console.print(f"[dim]Processing in batches of {batch_size}...[/dim]")

        first_batch = documents[:batch_size]
        vector_store = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(persist_dir),
        )

        remaining = documents[batch_size:]
        total_batches = (len(remaining) + batch_size - 1) // batch_size

        for i in tqdm(
            range(0, len(remaining), batch_size),
            desc="Embedding batches",
            unit="batch",
            total=total_batches,
        ):
            batch = remaining[i : i + batch_size]
            vector_store.add_documents(batch)

    console.print(f"[bold green]Vector store created with {len(documents)} documents[/bold green]")

    return vector_store


def load_vector_store(
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
) -> Chroma:
    """Load an existing ChromaDB vector store (cached after first load)."""
    cache_key = (collection_name, str(persist_directory))
    if cache_key in _vector_store_cache:
        return _vector_store_cache[cache_key]

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

    count = vector_store._collection.count()
    console.print(f"[bold green]Loaded vector store with {count} documents[/bold green]")

    _vector_store_cache[cache_key] = vector_store
    return vector_store


def delete_vector_store(
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
) -> bool:
    """Delete the vector store directory. Returns True if deleted, False if not found."""
    persist_dir = Path(persist_directory)

    if persist_dir.exists():
        shutil.rmtree(persist_dir)
        console.print(f"[bold yellow]Deleted vector store at {persist_dir}[/bold yellow]")
        return True

    console.print(f"[dim]No vector store found at {persist_dir}[/dim]")
    return False


def _make_doc_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Create a RecursiveCharacterTextSplitter tuned for documentation Markdown."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n## ",  # Markdown H2
            "\n### ",  # Markdown H3
            "\n#### ",  # Markdown H4
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentences
            " ",  # Words
            "",  # Characters
        ],
        is_separator_regex=False,
    )


def create_parent_retriever(
    documents: list[Document],
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
    parent_store_dir: Path | str = config.PARENT_STORE_DIR,
) -> ParentDocumentRetriever:
    """Create a ParentDocumentRetriever and ingest documents.

    Two-level indexing:
    - Child chunks (CHILD_CHUNK_SIZE chars) are embedded in Chroma for precise retrieval.
    - Parent sections (PARENT_CHUNK_SIZE chars) are stored in LocalFileStore on disk.

    At query time the retriever finds matching child chunks, then returns their
    full parent sections as context — giving the LLM complete, coherent content.

    Args:
        documents: Full-page Documents (one per scraped page, no pre-chunking)
        collection_name: Name for the ChromaDB child-chunk collection
        persist_directory: Directory for the ChromaDB vector store
        parent_store_dir: Directory for LocalFileStore (parent sections)

    Returns:
        Configured ParentDocumentRetriever (already populated)
    """
    persist_dir = Path(persist_directory)
    store_dir = Path(parent_store_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()

    # Small child chunks: precise semantic matching in vector space
    child_splitter = _make_doc_splitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
    )
    # Large parent sections: rich context returned to the LLM
    parent_splitter = _make_doc_splitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    docstore = create_kv_docstore(LocalFileStore(str(store_dir)))

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    console.print(
        f"[bold blue]Building parent-child index for {len(documents)} pages...[/bold blue]"
    )
    console.print(f"[dim]Child chunks: {config.CHILD_CHUNK_SIZE} chars → Chroma embeddings[/dim]")
    console.print(f"[dim]Parent sections: {config.PARENT_CHUNK_SIZE} chars → LocalFileStore[/dim]")

    batch_size = 50
    for i in tqdm(
        range(0, len(documents), batch_size),
        desc="Indexing pages",
        unit="batch",
        total=(len(documents) + batch_size - 1) // batch_size,
    ):
        retriever.add_documents(documents[i : i + batch_size])

    child_count = vectorstore._collection.count()
    console.print(
        f"[bold green]ParentDocumentRetriever ready: "
        f"{len(documents)} pages → {child_count} child chunks[/bold green]"
    )
    return retriever


def load_parent_retriever(
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
    parent_store_dir: Path | str = config.PARENT_STORE_DIR,
) -> ParentDocumentRetriever:
    """Load an existing ParentDocumentRetriever from disk (cached after first load)."""
    cache_key = (collection_name, str(persist_directory), str(parent_store_dir))
    if cache_key in _retriever_cache:
        return _retriever_cache[cache_key]

    persist_dir = Path(persist_directory)
    store_dir = Path(parent_store_dir)

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. Run 'ingest' first to build the index."
        )
    if not store_dir.exists():
        raise FileNotFoundError(
            f"Parent document store not found at {store_dir}. "
            "Run 'ingest' first to build the index."
        )

    embeddings = get_embeddings()

    child_splitter = _make_doc_splitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
    )
    parent_splitter = _make_doc_splitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    docstore = create_kv_docstore(LocalFileStore(str(store_dir)))

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    child_count = vectorstore._collection.count()
    console.print(
        f"[bold green]Loaded ParentDocumentRetriever ({child_count} child chunks)[/bold green]"
    )

    _retriever_cache[cache_key] = retriever
    return retriever
