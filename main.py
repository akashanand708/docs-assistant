#!/usr/bin/env python3
"""CLI interface for PingIdentity Documentation Assistant."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from rag import config

app = typer.Typer(
    name="pingid-assistant",
    help="AI assistant for PingIdentity documentation",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    url: str = typer.Option(
        config.DOCS_BASE_URL,
        "--url",
        "-u",
        help="Base URL of the documentation site to scrape",
    ),
    max_pages: int = typer.Option(
        config.MAX_PAGES,
        "--max-pages",
        "-m",
        help="Maximum number of pages to scrape",
    ),
    chunk_size: int = typer.Option(
        config.CHUNK_SIZE,
        "--chunk-size",
        help="Size of text chunks in characters",
    ),
    chunk_overlap: int = typer.Option(
        config.CHUNK_OVERLAP,
        "--chunk-overlap",
        help="Overlap between chunks in characters",
    ),
    collection: str = typer.Option(
        config.COLLECTION_NAME,
        "--collection",
        "-c",
        help="ChromaDB collection name",
    ),
    persist_dir: Path = typer.Option(
        config.CHROMA_PERSIST_DIR,
        "--persist-dir",
        "-d",
        help="Directory to persist the vector database",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Delete existing vector store before ingesting",
    ),
) -> None:
    """Scrape documentation and ingest into vector database.

    This command crawls a documentation website starting from the base URL,
    extracts text content from each page, splits it into overlapping chunks,
    generates embeddings, and stores them in a ChromaDB vector database.

    Args:
        url: Base URL of the documentation site (e.g., https://docs.pingidentity.com)
        max_pages: Maximum number of pages to crawl (use -1 for unlimited)
        chunk_size: Size of text chunks in characters for splitting documents
        chunk_overlap: Number of overlapping characters between consecutive chunks
        collection: Name of the ChromaDB collection to store embeddings
        persist_dir: Local directory path for persisting the vector database
        force: If True, deletes existing vector store before ingesting

    Example:
        $ python main.py ingest --url https://docs.pingidentity.com --max-pages 100
        $ python main.py ingest --force  # Re-ingest from scratch
    """
    from rag.processor import process_scraped_pages
    from rag.scraper import DocsScraper
    from rag.vector_store import create_vector_store, delete_vector_store

    if force:
        delete_vector_store(persist_dir)

    console.print("[bold]Step 1/3: Scraping documentation...[/bold]")
    scraper = DocsScraper(base_url=url, max_pages=max_pages)
    pages = list(scraper.crawl())

    if not pages:
        console.print("[red]No pages scraped. Check the URL and try again.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Step 2/3: Processing {len(pages)} pages into chunks...[/bold]")
    documents = process_scraped_pages(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Save chunks to file for recovery
    import json

    persist_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    chunks_file = persist_dir / "pending_chunks.json"
    chunks_data = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
    with open(chunks_file, "w") as f:
        json.dump(chunks_data, f)
    console.print(f"[dim]Saved {len(documents)} chunks to {chunks_file}[/dim]")

    console.print("\n[bold]Step 3/3: Creating vector embeddings...[/bold]")
    create_vector_store(
        documents=documents,
        collection_name=collection,
        persist_directory=persist_dir,
    )

    # Remove pending chunks file on success
    chunks_file.unlink(missing_ok=True)

    console.print("\n[bold green]✓ Ingestion complete![/bold green]")
    console.print(f"  Pages scraped: {len(pages)}")
    console.print(f"  Document chunks: {len(documents)}")
    console.print(f"  Vector store: {persist_dir}")


@app.command()
def embed(
    collection: str = typer.Option(
        config.COLLECTION_NAME,
        "--collection",
        "-c",
        help="ChromaDB collection name",
    ),
    persist_dir: Path = typer.Option(
        config.CHROMA_PERSIST_DIR,
        "--persist-dir",
        "-d",
        help="Directory where vector database is persisted",
    ),
    chunks_file: Path = typer.Option(
        None,
        "--chunks-file",
        "-f",
        help="Path to chunks JSON file (default: <persist-dir>/pending_chunks.json)",
    ),
) -> None:
    """Resume embedding from saved chunks file.

    Use this command to retry embedding after a failure during the ingest process.
    When ingest fails partway through (e.g., due to API rate limits), it saves
    processed chunks to a JSON file. This command resumes from that checkpoint.

    Args:
        collection: Name of the ChromaDB collection to store embeddings
        persist_dir: Directory where vector database is persisted
        chunks_file: Path to JSON file containing chunks (default: <persist-dir>/pending_chunks.json)

    Example:
        $ python main.py embed  # Resume from default pending_chunks.json
        $ python main.py embed --chunks-file ./my_chunks.json
    """
    import json

    from langchain_core.documents import Document

    from rag.vector_store import create_vector_store

    # Default to pending_chunks.json in persist directory
    if chunks_file is None:
        chunks_file = persist_dir / "pending_chunks.json"

    if not chunks_file.exists():
        console.print(f"[red]Chunks file not found: {chunks_file}[/red]")
        console.print("[dim]Run 'ingest' first, or specify --chunks-file[/dim]")
        raise typer.Exit(1)

    console.print(f"[bold]Loading chunks from {chunks_file}...[/bold]")
    with open(chunks_file) as f:
        chunks_data = json.load(f)

    documents = [
        Document(page_content=chunk["content"], metadata=chunk["metadata"]) for chunk in chunks_data
    ]
    console.print(f"[green]Loaded {len(documents)} chunks[/green]")

    console.print("\n[bold]Creating vector embeddings...[/bold]")
    create_vector_store(
        documents=documents,
        collection_name=collection,
        persist_directory=persist_dir,
    )

    # Remove pending chunks file on success
    chunks_file.unlink(missing_ok=True)

    console.print("\n[bold green]✓ Embedding complete![/bold green]")
    console.print(f"  Document chunks: {len(documents)}")
    console.print(f"  Vector store: {persist_dir}")


@app.command()
def chat(
    collection: str = typer.Option(
        config.COLLECTION_NAME,
        "--collection",
        "-c",
        help="ChromaDB collection name",
    ),
    persist_dir: Path = typer.Option(
        config.CHROMA_PERSIST_DIR,
        "--persist-dir",
        "-d",
        help="Directory where vector database is persisted",
    ),
    show_sources: bool = typer.Option(
        False,
        "--sources",
        "-s",
        help="Display source documents with responses",
    ),
) -> None:
    """Start interactive chat with the documentation assistant.

    Launches an interactive terminal-based chat session where you can ask
    questions about PingIdentity documentation. Uses RAG (Retrieval-Augmented
    Generation) to find relevant documentation and generate accurate answers.

    Args:
        collection: Name of the ChromaDB collection containing embeddings
        persist_dir: Directory where vector database is persisted
        show_sources: If True, displays source URLs for retrieved documents

    Example:
        $ python main.py chat
        $ python main.py chat --sources  # Show source documents
    """
    from rag.rag_chain import interactive_chat

    interactive_chat(
        collection_name=collection,
        persist_directory=persist_dir,
        show_sources=show_sources,
    )


@app.command()
def web(
    collection: str = typer.Option(
        config.COLLECTION_NAME,
        "--collection",
        "-c",
        help="ChromaDB collection name",
    ),
    persist_dir: Path = typer.Option(
        config.CHROMA_PERSIST_DIR,
        "--persist-dir",
        "-d",
        help="Directory where vector database is persisted",
    ),
    port: int = typer.Option(
        7860,
        "--port",
        "-p",
        help="Port to run the web server on",
    ),
    share: bool = typer.Option(
        False,
        "--share",
        help="Create a public Gradio share link (valid for 72 hours)",
    ),
    show_sources: bool = typer.Option(
        False,
        "--sources",
        "-s",
        help="Show source documents with URLs after each response",
    ),
) -> None:
    """Launch web UI for the documentation assistant.

    Starts a Gradio-based web interface for chatting with the documentation
    assistant. Supports streaming responses and optional source citations.

    Args:
        collection: Name of the ChromaDB collection containing embeddings
        persist_dir: Directory where vector database is persisted
        port: Port number for the web server (default: 7860)
        share: If True, creates a public ngrok tunnel URL (valid for 72 hours)
        show_sources: If True, displays source document URLs after each response

    Example:
        $ python main.py web
        $ python main.py web --share --sources  # Public URL with sources
        $ python main.py web --port 8080
    """
    from rag.web_ui import launch_web_ui

    launch_web_ui(
        collection_name=collection,
        persist_directory=persist_dir,
        share=share,
        server_port=port,
        show_sources=show_sources,
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    collection: str = typer.Option(
        config.COLLECTION_NAME,
        "--collection",
        "-c",
        help="ChromaDB collection name",
    ),
    persist_dir: Path = typer.Option(
        config.CHROMA_PERSIST_DIR,
        "--persist-dir",
        "-d",
        help="Directory where vector database is persisted",
    ),
    show_sources: bool = typer.Option(
        False,
        "--sources",
        "-s",
        help="Display source documents with the response",
    ),
) -> None:
    """Ask a single question and get an answer.

    Performs a one-shot RAG query against the documentation. Retrieves relevant
    documents from the vector store and generates an answer using the LLM.

    Args:
        question: The question to ask about PingIdentity documentation
        collection: Name of the ChromaDB collection containing embeddings
        persist_dir: Directory where vector database is persisted
        show_sources: If True, displays source URLs for retrieved documents

    Example:
        $ python main.py ask "How do I configure MFA?"
        $ python main.py ask "What is PingFederate?" --sources
    """
    from rich.markdown import Markdown

    from rag.rag_chain import query

    result = query(
        question=question,
        collection_name=collection,
        persist_directory=persist_dir,
    )

    console.print(Markdown(result["answer"]))

    if show_sources and result.get("context"):
        console.print("\n[dim]Sources:[/dim]")
        seen_sources = set()
        for doc in result["context"]:
            source = doc.metadata.get("source", "Unknown")
            if source not in seen_sources:
                seen_sources.add(source)
                title = doc.metadata.get("title", "")
                console.print(f"  [dim]• {title}[/dim]")
                console.print(f"    {source}")


@app.command()
def search(
    query_text: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(
        config.TOP_K_RESULTS,
        "--top-k",
        "-k",
        help="Number of results to return",
    ),
    collection: str = typer.Option(
        config.COLLECTION_NAME,
        "--collection",
        "-c",
        help="ChromaDB collection name",
    ),
    persist_dir: Path = typer.Option(
        config.CHROMA_PERSIST_DIR,
        "--persist-dir",
        "-d",
        help="Directory where vector database is persisted",
    ),
) -> None:
    """Search the vector database for similar documents.

    Performs a semantic similarity search in the vector database without
    generating an LLM response. Useful for debugging or exploring what
    documents are retrieved for a given query.

    Args:
        query_text: The search query to find similar documents
        top_k: Number of most similar results to return
        collection: Name of the ChromaDB collection to search
        persist_dir: Directory where vector database is persisted

    Example:
        $ python main.py search "MFA configuration"
        $ python main.py search "OAuth setup" --top-k 10
    """
    from rag.vector_store import similarity_search

    results = similarity_search(
        query=query_text,
        k=top_k,
        collection_name=collection,
        persist_directory=persist_dir,
    )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"[bold]Found {len(results)} results:[/bold]\n")

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "Untitled")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        total_chunks = doc.metadata.get("total_chunks", 1)

        console.print(f"[bold cyan]{i}. {title}[/bold cyan]")
        console.print(f"   [dim]Chunk {chunk_idx + 1}/{total_chunks} | {source}[/dim]")
        # Show preview of content
        preview = doc.page_content[:200].replace("\n", " ")
        if len(doc.page_content) > 200:
            preview += "..."
        console.print(f"   {preview}\n")


@app.command()
def status(
    collection: str = typer.Option(
        config.COLLECTION_NAME,
        "--collection",
        "-c",
        help="ChromaDB collection name",
    ),
    persist_dir: Path = typer.Option(
        config.CHROMA_PERSIST_DIR,
        "--persist-dir",
        "-d",
        help="Directory where vector database is persisted",
    ),
) -> None:
    """Show status of the vector database.

    Displays information about the current vector database including
    collection name, storage location, document count, and embedding model.

    Args:
        collection: Name of the ChromaDB collection to check
        persist_dir: Directory where vector database is persisted

    Example:
        $ python main.py status
    """
    from rag.vector_store import load_vector_store

    try:
        vector_store = load_vector_store(collection, persist_dir)
        collection_obj = vector_store._collection
        count = collection_obj.count()

        console.print("[bold]Vector Database Status[/bold]")
        console.print(f"  Collection: {collection}")
        console.print(f"  Location: {persist_dir}")
        console.print(f"  Documents: {count}")
        if config.USE_LOCAL_EMBEDDINGS:
            console.print(f"  Embedding: local ({config.LOCAL_EMBEDDING_MODEL})")
        else:
            console.print(f"  Embedding: API ({config.EMBEDDING_MODEL})")
    except FileNotFoundError:
        console.print(f"[yellow]No vector store found at {persist_dir}[/yellow]")
        console.print("[dim]Run 'ingest' command to create one.[/dim]")


@app.command()
def clear(
    persist_dir: Path = typer.Option(
        config.CHROMA_PERSIST_DIR,
        "--persist-dir",
        "-d",
        help="Directory where vector database is persisted",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Delete the vector database.

    Removes all data from the vector database. This operation is irreversible.
    You will need to run 'ingest' again to rebuild the database.

    Args:
        persist_dir: Directory where vector database is persisted
        force: If True, skips the confirmation prompt

    Example:
        $ python main.py clear
        $ python main.py clear --force  # Skip confirmation
    """
    from rag.vector_store import delete_vector_store

    if not force:
        confirm = typer.confirm(f"Delete vector store at {persist_dir}?")
        if not confirm:
            raise typer.Abort()

    delete_vector_store(persist_dir)


if __name__ == "__main__":
    app()
