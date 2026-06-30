"""RAG chain for querying the documentation."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Generator
from pathlib import Path
from typing import Any

import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel

from .. import config
from ..ingestion.vector_store import load_parent_retriever, load_vector_store

console = Console()

# ---------------------------------------------------------------------------
# Cross-encoder reranker (loaded once per process)
# ---------------------------------------------------------------------------

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            console.print("[dim]Cross-encoder reranker loaded[/dim]")
        except ImportError:
            console.print(
                "[yellow]sentence-transformers not installed — skipping reranker[/yellow]"
            )
    return _reranker


def _rerank(question: str, docs: list, top_k: int) -> list:
    reranker = _get_reranker()
    if reranker is None or len(docs) <= top_k:
        return docs[:top_k]
    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs, strict=False), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


SYSTEM_PROMPT = """You are a helpful assistant for {site_name} documentation. Answer questions directly and accurately using only the documentation context below.

- Include the full documentation URL when referencing information
- Include configuration steps, code examples, or commands when relevant
- End with a "References:" section listing relevant URLs
- If the answer is not in the provided context, say so in one sentence

Documentation context:
{context}
"""


# ---------------------------------------------------------------------------
# Retriever helpers
# ---------------------------------------------------------------------------


def _load_retriever(
    collection_name: str,
    persist_directory: Path | str,
    top_k: int,
):
    """Return the configured retriever (uses module-level cache)."""
    if config.USE_PARENT_RETRIEVER:
        retriever = load_parent_retriever(collection_name, persist_directory)
        retriever.search_kwargs = {"k": top_k * 2}
    else:
        vector_store = load_vector_store(collection_name, persist_directory)
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k * 2 if config.USE_PAGE_INDEX else top_k,
                "fetch_k": config.RETRIEVE_K * 2 if config.USE_PAGE_INDEX else config.RETRIEVE_K,
                "lambda_mult": 0.7,
            },
        )
    return retriever


def _format_docs(docs: list) -> str:
    """Format retrieved documents into a context string for the LLM."""
    if not config.USE_PARENT_RETRIEVER and config.USE_PAGE_INDEX:
        return _format_docs_by_page(docs)

    formatted = []
    total_chars = 0
    max_chars = (
        config.PARENT_CONTEXT_CHARS if config.USE_PARENT_RETRIEVER else config.MAX_CONTEXT_CHARS
    )

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "Untitled")
        content = doc.page_content

        remaining = max_chars - total_chars - 100
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:remaining] + "..."

        entry = f"[Doc {i}] {title}\nURL: {source}\nContent: {content}"
        formatted.append(entry)
        total_chars += len(entry)

        if total_chars >= max_chars:
            break

    return "\n\n---\n\n".join(formatted)


def _format_docs_by_page(docs: list) -> str:
    """Format documents grouped by source URL for cleaner citations."""
    pages: OrderedDict[str, dict] = OrderedDict()

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "Untitled")
        chunk_index = doc.metadata.get("chunk_index", 0)

        if source not in pages:
            pages[source] = {"title": title, "chunks": []}
        pages[source]["chunks"].append((chunk_index, doc.page_content))

    formatted = []
    total_chars = 0
    max_chars = config.MAX_CONTEXT_CHARS

    for i, (source, page_data) in enumerate(pages.items(), 1):
        if total_chars >= max_chars:
            break

        sorted_chunks = sorted(page_data["chunks"], key=lambda x: x[0])
        combined = "\n\n".join(chunk for _, chunk in sorted_chunks)

        remaining = min(config.PAGE_CONTEXT_CHARS, max_chars - total_chars - 100)
        if remaining <= 0:
            break
        if len(combined) > remaining:
            combined = combined[:remaining] + "..."

        entry = f"[Page {i}] {page_data['title']}\nURL: {source}\nContent:\n{combined}"
        formatted.append(entry)
        total_chars += len(entry)

    return "\n\n" + "=" * 50 + "\n\n".join(formatted)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _build_llm(model_name: str, temperature: float) -> ChatBedrockConverse:
    """Build a ChatBedrockConverse client, popping AWS_PROFILE so boto3 uses bearer auth."""
    saved_profile = os.environ.pop("AWS_PROFILE", None)
    try:
        region = os.environ.get("AWS_REGION", "us-east-1")
        boto_client = boto3.Session(region_name=region).client(
            "bedrock-runtime", region_name=region
        )
        return ChatBedrockConverse(
            client=boto_client,
            model=model_name,
            temperature=temperature,
        )
    finally:
        if saved_profile is not None:
            os.environ["AWS_PROFILE"] = saved_profile


def _build_messages(
    question: str,
    context: str,
    chat_history: list[tuple[str, str]] | None,
) -> list:
    system_text = SYSTEM_PROMPT.format(site_name=config.SITE_NAME, context=context)
    messages: list = [SystemMessage(content=system_text)]
    if chat_history:
        for human_msg, ai_msg in chat_history:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))
    messages.append(HumanMessage(content=question))
    return messages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def query(
    question: str,
    chat_history: list[tuple[str, str]] | None = None,
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
    top_k: int = config.TOP_K_RESULTS,
    model_name: str = config.LLM_MODEL,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Query the RAG system and return the full answer (non-streaming)."""
    retriever = _load_retriever(collection_name, persist_directory, top_k)
    docs = retriever.invoke(question)
    docs = _rerank(question, docs, top_k)
    context = _format_docs(docs)
    llm = _build_llm(model_name, temperature)
    messages = _build_messages(question, context, chat_history)
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return {"answer": content, "context": docs}


def query_stream(
    question: str,
    chat_history: list[tuple[str, str]] | None = None,
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
    top_k: int = config.TOP_K_RESULTS,
    model_name: str = config.LLM_MODEL,
    temperature: float = 0.1,
) -> Generator[str, None, None]:
    """Stream the response token-by-token via Bedrock.

    First token arrives in ~2s. Used by the web UI for low perceived latency.
    """
    retriever = _load_retriever(collection_name, persist_directory, top_k)
    docs = retriever.invoke(question)
    docs = _rerank(question, docs, top_k)
    context = _format_docs(docs)
    llm = _build_llm(model_name, temperature)
    messages = _build_messages(question, context, chat_history)
    try:
        for chunk in llm.stream(messages):
            content = chunk.content
            if isinstance(content, list):
                text = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                )
            else:
                text = content
            if text:
                yield text
    except Exception as e:
        yield f"\n\n⚠️ Error: {e}"


def interactive_chat(
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
    show_sources: bool = False,
) -> None:
    """Start an interactive chat session in the terminal."""
    # Preload retriever once so first question doesn't pay the load cost.
    console.print("[dim]Loading index...[/dim]")
    _load_retriever(collection_name, persist_directory, config.TOP_K_RESULTS)

    console.print(
        Panel.fit(
            f"[bold blue]{config.SITE_NAME} Documentation Assistant[/bold blue]\n"
            f"[dim]Ask questions about {config.SITE_NAME} documentation.\n"
            "Type 'quit' or 'exit' to end the session.\n"
            "Type 'clear' to reset chat history.\n"
            "Type 'sources' to toggle source display.[/dim]",
            border_style="blue",
        )
    )

    chat_history: list[tuple[str, str]] = []

    while True:
        try:
            console.print()
            user_input = console.input("[bold green]You:[/bold green] ").strip()

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break
            if user_input.lower() == "clear":
                chat_history = []
                console.print("[dim]Chat history cleared.[/dim]")
                continue
            if user_input.lower() == "sources":
                show_sources = not show_sources
                console.print(f"[dim]Source display: {'on' if show_sources else 'off'}[/dim]")
                continue

            console.print()

            stream = query_stream(
                question=user_input,
                chat_history=chat_history,
                collection_name=collection_name,
                persist_directory=persist_directory,
            )

            # Show spinner during retrieval + LLM warm-up (before first token).
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                first_chunk = next(stream, None)

            full_answer = ""
            source_docs = []
            console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
            if first_chunk is not None:
                print(first_chunk, end="", flush=True)
                full_answer = first_chunk
                for chunk in stream:
                    print(chunk, end="", flush=True)
                    full_answer += chunk

            print()  # newline after streamed output

            if show_sources:
                retriever = _load_retriever(
                    collection_name, persist_directory, config.TOP_K_RESULTS
                )
                source_docs = retriever.invoke(user_input)

            if show_sources and source_docs:
                console.print()
                console.print("[dim]Sources:[/dim]")
                seen_sources: set[str] = set()
                for doc in source_docs:
                    source = doc.metadata.get("source", "Unknown")
                    if source not in seen_sources:
                        seen_sources.add(source)
                        title = doc.metadata.get("title", "")
                        console.print(f"  [dim]• {title}[/dim]")
                        console.print(f"    [link={source}]{source}[/link]")

            chat_history.append((user_input, full_answer))

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
