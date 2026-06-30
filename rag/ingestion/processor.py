"""Document processing and chunking for RAG pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

from .. import config

if TYPE_CHECKING:
    from .scraper import ScrapedPage

console = Console()


def create_text_splitter(
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Create a text splitter optimized for documentation content."""
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


def process_scraped_pages(
    pages: list[ScrapedPage],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Process scraped pages into LangChain Documents with chunking.

    Args:
        pages: List of scraped pages from the web scraper
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of LangChain Document objects ready for embedding
    """
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    documents: list[Document] = []

    console.print(f"[bold blue]Processing {len(pages)} pages into chunks...[/bold blue]")

    for page in pages:
        # Create full document text with metadata context
        full_text = page.to_document_text()

        # Create metadata for this page
        metadata = {
            "source": page.url,
            "title": page.title,
            "breadcrumbs": " > ".join(page.breadcrumbs) if page.breadcrumbs else "",
        }

        # Split into chunks
        chunks = text_splitter.split_text(full_text)

        # Create Document objects for each chunk
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            documents.append(Document(page_content=chunk, metadata=doc_metadata))

    console.print(
        f"[bold green]Created {len(documents)} document chunks from {len(pages)} pages[/bold green]"
    )

    return documents


def pages_to_documents(pages: list[ScrapedPage]) -> list[Document]:
    """Convert scraped pages to full-page Documents without chunking.

    Used as input to ParentDocumentRetriever. The retriever itself handles
    splitting into parent sections and child chunks internally.

    Args:
        pages: List of scraped pages from the web scraper

    Returns:
        List of LangChain Document objects — one per page, full content preserved
    """
    documents: list[Document] = []

    console.print(f"[bold blue]Converting {len(pages)} pages to documents...[/bold blue]")

    for page in pages:
        full_text = page.to_document_text()
        metadata = {
            "source": page.url,
            "title": page.title,
            "breadcrumbs": " > ".join(page.breadcrumbs) if page.breadcrumbs else "",
        }
        documents.append(Document(page_content=full_text, metadata=metadata))

    console.print(f"[bold green]Created {len(documents)} page documents (no chunking)[/bold green]")
    return documents
