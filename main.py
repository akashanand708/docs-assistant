#!/usr/bin/env python3
"""CLI interface for Documentation Assistant."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from rag import config

app = typer.Typer(
    name="docs-assistant",
    help="AI assistant for documentation",
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
        url: Base URL of the documentation site (e.g., https://docs.example.com)
        max_pages: Maximum number of pages to crawl (use -1 for unlimited)
        chunk_size: Size of text chunks in characters for splitting documents
        chunk_overlap: Number of overlapping characters between consecutive chunks
        collection: Name of the ChromaDB collection to store embeddings
        persist_dir: Local directory path for persisting the vector database
        force: If True, deletes existing vector store before ingesting

    Example:
        $ python main.py ingest --url https://docs.example.com --max-pages 100
        $ python main.py ingest --force  # Re-ingest from scratch
    """
    import json

    from rag.ingestion.scraper import DocsScraper
    from rag.ingestion.vector_store import delete_vector_store

    if force:
        delete_vector_store(persist_dir)

    console.print("[bold]Step 1/3: Scraping documentation...[/bold]")
    scraper = DocsScraper(base_url=url, max_pages=max_pages)
    pages = list(scraper.crawl())

    if not pages:
        console.print("[red]No pages scraped. Check the URL and try again.[/red]")
        raise typer.Exit(1)

    persist_dir.mkdir(parents=True, exist_ok=True)

    if config.USE_PARENT_RETRIEVER:
        from rag.ingestion.processor import pages_to_documents
        from rag.ingestion.vector_store import create_parent_retriever

        console.print(f"\n[bold]Step 2/3: Converting {len(pages)} pages to documents...[/bold]")
        documents = pages_to_documents(pages)

        # Save for recovery in case indexing is interrupted
        pages_file = persist_dir / "pending_pages.json"
        pages_data = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
        with open(pages_file, "w") as f:
            json.dump(pages_data, f)
        console.print(f"[dim]Saved {len(documents)} page documents to {pages_file}[/dim]")

        console.print("\n[bold]Step 3/3: Building parent-child index...[/bold]")
        create_parent_retriever(
            documents=documents,
            collection_name=collection,
            persist_directory=persist_dir,
        )
        pages_file.unlink(missing_ok=True)

        console.print("\n[bold green]✓ Ingestion complete![/bold green]")
        console.print(f"  Pages scraped:    {len(pages)}")
        console.print("  Index type:       ParentDocumentRetriever")
        console.print(f"  Vector store:     {persist_dir}")
        console.print(f"  Parent store:     {config.PARENT_STORE_DIR}")
    else:
        from rag.ingestion.processor import process_scraped_pages
        from rag.ingestion.vector_store import create_vector_store

        console.print(f"\n[bold]Step 2/3: Processing {len(pages)} pages into chunks...[/bold]")
        documents = process_scraped_pages(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Save chunks for recovery
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
        chunks_file.unlink(missing_ok=True)

        console.print("\n[bold green]✓ Ingestion complete![/bold green]")
        console.print(f"  Pages scraped:    {len(pages)}")
        console.print(f"  Document chunks:  {len(documents)}")
        console.print(f"  Vector store:     {persist_dir}")

    from rag.evaluator import generate_testset

    generate_testset(
        collection_name=collection,
        persist_directory=persist_dir,
    )


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

    if config.USE_PARENT_RETRIEVER:
        from rag.ingestion.vector_store import create_parent_retriever

        # Default to pending_pages.json in parent retriever mode
        if chunks_file is None:
            chunks_file = persist_dir / "pending_pages.json"

        if not chunks_file.exists():
            console.print(f"[red]Pages file not found: {chunks_file}[/red]")
            console.print("[dim]Run 'ingest' first, or specify --chunks-file[/dim]")
            raise typer.Exit(1)

        console.print(f"[bold]Loading page documents from {chunks_file}...[/bold]")
        with open(chunks_file) as f:
            pages_data = json.load(f)

        documents = [
            Document(page_content=p["content"], metadata=p["metadata"]) for p in pages_data
        ]
        console.print(f"[green]Loaded {len(documents)} page documents[/green]")

        console.print("\n[bold]Building parent-child index...[/bold]")
        create_parent_retriever(
            documents=documents,
            collection_name=collection,
            persist_directory=persist_dir,
        )
        chunks_file.unlink(missing_ok=True)

        console.print("\n[bold green]✓ Indexing complete![/bold green]")
        console.print(f"  Page documents:  {len(documents)}")
        console.print(f"  Parent store:    {config.PARENT_STORE_DIR}")
    else:
        from rag.ingestion.vector_store import create_vector_store

        # Default to pending_chunks.json in flat chunk mode
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
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in chunks_data
        ]
        console.print(f"[green]Loaded {len(documents)} chunks[/green]")

        console.print("\n[bold]Creating vector embeddings...[/bold]")
        create_vector_store(
            documents=documents,
            collection_name=collection,
            persist_directory=persist_dir,
        )
        chunks_file.unlink(missing_ok=True)

        console.print("\n[bold green]✓ Embedding complete![/bold green]")
        console.print(f"  Document chunks: {len(documents)}")
        console.print(f"  Vector store:    {persist_dir}")


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
    questions about the documentation. Uses RAG (Retrieval-Augmented
    Generation) to find relevant documentation and generate accurate answers.

    Args:
        collection: Name of the ChromaDB collection containing embeddings
        persist_dir: Directory where vector database is persisted
        show_sources: If True, displays source URLs for retrieved documents

    Example:
        $ python main.py chat
        $ python main.py chat --sources  # Show source documents
    """
    from rag.query.rag_chain import interactive_chat

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
    root_path: str = typer.Option(
        "",
        "--root-path",
        help="Public base URL when behind a proxy/tunnel (e.g. ngrok URL)",
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
    from rag.query.web_ui import launch_web_ui

    launch_web_ui(
        collection_name=collection,
        persist_directory=persist_dir,
        share=share,
        server_port=port,
        show_sources=show_sources,
        root_path=root_path,
    )


@app.command(name="gen-testset")
def gen_testset(
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
    output: Path = typer.Option(
        Path("eval_testset.json"),
        "--output",
        "-o",
        help="Output path for the generated test set",
    ),
    n_pages: int = typer.Option(
        20,
        "--pages",
        "-p",
        help="Number of pages to sample from the index",
    ),
    questions_per_page: int = typer.Option(
        2,
        "--questions",
        "-q",
        help="Questions to generate per page",
    ),
) -> None:
    """Generate an evaluation test set from the existing vector index."""
    from rag.evaluator import generate_testset

    generate_testset(
        collection_name=collection,
        persist_directory=persist_dir,
        output_path=output,
        n_pages=n_pages,
        questions_per_page=questions_per_page,
    )


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
    from rag.ingestion.vector_store import load_vector_store

    try:
        vector_store = load_vector_store(collection, persist_dir)
        collection_obj = vector_store._collection
        count = collection_obj.count()

        console.print("[bold]Vector Database Status[/bold]")
        console.print(f"  Collection: {collection}")
        console.print(f"  Location: {persist_dir}")
        console.print(f"  Documents: {count}")
        console.print(f"  Embedding: local ({config.LOCAL_EMBEDDING_MODEL})")
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
    from rag.ingestion.vector_store import delete_vector_store

    if not force:
        confirm = typer.confirm(f"Delete vector store at {persist_dir}?")
        if not confirm:
            raise typer.Abort()

    delete_vector_store(persist_dir)


@app.command()
def evaluate(
    testset: Path = typer.Option(
        Path("eval_testset.json"),
        "--testset",
        "-t",
        help="Path to JSON test set file",
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
    top_k: int = typer.Option(
        config.TOP_K_RESULTS,
        "--top-k",
        "-k",
        help="Number of documents to retrieve per question",
    ),
    ragas: bool = typer.Option(
        False,
        "--ragas",
        help="Also run RAGAS quality metrics (needs ground_truth in test set and pip install ragas)",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Save results to this JSON file (default: eval_results.json next to testset)",
    ),
) -> None:
    """Benchmark the RAG system accuracy.

    Runs two benchmarks against a JSON test set:

    \b
    1. Hit Rate @k  (always)
       Checks whether the correct documentation page appeared anywhere in the
       top-k retrieved results.  Requires 'expected_url' in test cases.
       → Hit Rate: fraction of questions answered correctly
       → MRR:      mean reciprocal rank (rewards finding the right page higher)

    \b
    2. RAGAS metrics (only with --ragas flag)
       Uses an LLM as judge to score generation quality.
       Requires 'ground_truth' answers in test cases AND pip install ragas.
       Metrics: context_precision, context_recall, faithfulness, answer_relevancy

    \b
    Test set format (JSON array):
      [
        {
          "question":     "How do I configure OIDC?",
          "expected_url": "oidc",    // substring match
          "ground_truth": "To configure OIDC..."  // for --ragas only
        }
      ]

    \b
    Examples:
      python main.py evaluate                          # hit rate only (fast)
      python main.py evaluate --ragas                  # + RAGAS (slow, uses LLM)
      python main.py evaluate --testset my_tests.json  # custom test set
      python main.py evaluate --top-k 10               # test with k=10
    """
    from rag.evaluator import (
        load_testset,
        print_hit_rate_table,
        print_ragas_table,
        run_hit_rate,
        run_ragas,
        save_results,
    )

    if not testset.exists():
        console.print(f"[red]Test set not found: {testset}[/red]")
        console.print(
            "[dim]Create a JSON file with questions and expected_url fields, "
            "or point --testset at an existing file.[/dim]"
        )
        raise typer.Exit(1)

    index_mode = "ParentDocumentRetriever" if config.USE_PARENT_RETRIEVER else "Flat chunks (MMR)"
    console.print("\n[bold]Documentation Assistant — Benchmark[/bold]")
    console.print(f"  Index mode:  {index_mode}")
    console.print(f"  Top-k:       {top_k}")
    console.print(f"  Test set:    {testset}")
    console.print(f"  RAGAS:       {'yes' if ragas else 'no (use --ragas to enable)'}")

    cases = load_testset(testset)

    # ── Benchmark 1: Hit Rate ────────────────────────────────────────────
    hit_rate_result = run_hit_rate(
        cases=cases,
        k=top_k,
        collection_name=collection,
        persist_directory=persist_dir,
    )
    print_hit_rate_table(hit_rate_result)

    # ── Benchmark 2: RAGAS ──────────────────────────────────────────────
    ragas_scores: dict = {}
    if ragas:
        ragas_scores = run_ragas(
            cases=cases,
            k=top_k,
            collection_name=collection,
            persist_directory=persist_dir,
        )
        print_ragas_table(ragas_scores)

    # ── Save results ─────────────────────────────────────────────────────
    if output is None:
        output = testset.parent / "eval_results.json"
    save_results(hit_rate_result, ragas_scores, output)


if __name__ == "__main__":
    app()
