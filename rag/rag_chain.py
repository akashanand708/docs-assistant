"""RAG chain for querying the documentation."""

from __future__ import annotations

import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import config
from .vector_store import load_vector_store

console = Console()

# Track last LLM request time for rate limiting
_last_llm_request_time: float = 0.0


def _rate_limit_wait():
    """Wait if needed to respect rate limits."""
    global _last_llm_request_time
    now = time.time()
    elapsed = now - _last_llm_request_time
    if elapsed < config.LLM_MIN_REQUEST_INTERVAL:
        time.sleep(config.LLM_MIN_REQUEST_INTERVAL - elapsed)
    _last_llm_request_time = time.time()


# System prompt for the RAG assistant
SYSTEM_PROMPT = """You are a helpful assistant for PingIdentity documentation. Answer questions using the documentation context provided below.

Guidelines:
1. Be accurate and provide detailed explanations based on the documentation
2. ALWAYS include the full documentation URL when referencing information
3. Include configuration steps, code examples, or commands when available
4. At the end, list relevant documentation URLs as "References:" section

When information is incomplete or missing:
- Clearly state what information IS available vs what's NOT in the current context
- Suggest what the user might search for to find the specific documentation
- Ask clarifying questions to better understand what they need (e.g., "Are you looking for X or Y?")
- If you find related content, explain how it connects to their question

Be conversational and helpful. If you can partially answer, do so and ask if they need more specific information.

Documentation context:
{context}
"""


def create_llm(
    model_name: str = config.LLM_MODEL,
    temperature: float = 0.1,
) -> ChatOpenAI:
    """Create LLM client using GitHub Models API (GITHUB_TOKEN).

    - GitHub token as API key
    - GitHub Models endpoint as base URL
    - OpenAI-compatible model names (e.g., openai/gpt-4.1)
    """
    if not config.GITHUB_TOKEN:
        raise ValueError("GitHub token required. Set GITHUB_TOKEN environment variable.")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=config.GITHUB_TOKEN,
        openai_api_base=config.GITHUB_MODELS_URL,
    )


def _format_docs(docs: list) -> str:
    """Format retrieved documents into a single context string.

    When USE_PAGE_INDEX is enabled, groups chunks by source URL to provide
    page-level context with better citations.

    Truncates to MAX_CONTEXT_CHARS to stay within GitHub Models token limit.
    """
    if config.USE_PAGE_INDEX:
        return _format_docs_by_page(docs)

    # Original chunk-based formatting
    formatted = []
    total_chars = 0
    max_chars = config.MAX_CONTEXT_CHARS

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "Untitled")
        content = doc.page_content

        # Truncate content if needed to stay within limit
        remaining = max_chars - total_chars - 100  # Reserve space for formatting
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
    """Format documents grouped by source URL (page-level indexing).

    Groups multiple chunks from the same page together, providing better
    context and cleaner citations.
    """
    from collections import OrderedDict

    # Group chunks by source URL, preserving order of first occurrence
    pages: OrderedDict[str, dict] = OrderedDict()

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "Untitled")
        chunk_index = doc.metadata.get("chunk_index", 0)
        content = doc.page_content

        if source not in pages:
            pages[source] = {
                "title": title,
                "chunks": [],
                "total_chars": 0,
            }

        # Add chunk with its index for ordering
        pages[source]["chunks"].append((chunk_index, content))

    # Format pages
    formatted = []
    total_chars = 0
    max_chars = config.MAX_CONTEXT_CHARS
    page_max = config.PAGE_CONTEXT_CHARS

    for i, (source, page_data) in enumerate(pages.items(), 1):
        if total_chars >= max_chars:
            break

        title = page_data["title"]
        # Sort chunks by index and combine
        sorted_chunks = sorted(page_data["chunks"], key=lambda x: x[0])
        combined_content = "\n\n".join(chunk for _, chunk in sorted_chunks)

        # Truncate page content
        remaining = min(page_max, max_chars - total_chars - 100)
        if remaining <= 0:
            break
        if len(combined_content) > remaining:
            combined_content = combined_content[:remaining] + "..."

        entry = f"[Page {i}] {title}\nURL: {source}\nContent:\n{combined_content}"
        formatted.append(entry)
        total_chars += len(entry)

    return "\n\n" + "=" * 50 + "\n\n".join(formatted)


def query(
    question: str,
    chat_history: list[tuple[str, str]] | None = None,
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
    top_k: int = config.TOP_K_RESULTS,
    model_name: str = config.LLM_MODEL,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Query the RAG system with a question.

    Args:
        question: User's question
        chat_history: Optional list of (user_msg, ai_msg) tuples for context
        collection_name: ChromaDB collection name
        persist_directory: Path to ChromaDB persistence directory
        top_k: Number of documents to retrieve
        model_name: LLM model to use
        temperature: LLM temperature

    Returns:
        Dictionary with 'answer' and 'context' keys
    """
    # Load vector store and create retriever with MMR for diverse results
    vector_store = load_vector_store(collection_name, persist_directory)

    # When using page index, fetch more chunks since they'll be grouped by URL
    effective_k = top_k * 2 if config.USE_PAGE_INDEX else top_k
    effective_fetch_k = config.RETRIEVE_K * 2 if config.USE_PAGE_INDEX else config.RETRIEVE_K

    retriever = vector_store.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diversity
        search_kwargs={
            "k": effective_k,  # Final number to return
            "fetch_k": effective_fetch_k,  # Candidates to consider
            "lambda_mult": 0.7,  # 0=max diversity, 1=max relevance
        },
    )

    # Create LLM
    llm = create_llm(model_name=model_name, temperature=temperature)

    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{question}"),
        ]
    )

    # Retrieve documents
    docs = retriever.invoke(question)
    context = _format_docs(docs)

    # Convert chat history to messages
    messages = []
    if chat_history:
        for human_msg, ai_msg in chat_history:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))

    # Build and invoke chain using LCEL
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": messages,
        }
    )

    return {
        "answer": answer,
        "context": docs,
    }


def query_stream(
    question: str,
    chat_history: list[tuple[str, str]] | None = None,
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
    top_k: int = config.TOP_K_RESULTS,
    model_name: str = config.LLM_MODEL,
    temperature: float = 0.1,
) -> Generator[str, None, None]:
    """Stream query responses for faster perceived response time.

    Includes rate limiting and retry logic for GitHub Models API.

    Yields:
        str: Chunks of the response as they're generated
    """
    # Load vector store and create retriever with MMR
    vector_store = load_vector_store(collection_name, persist_directory)

    # When using page index, fetch more chunks since they'll be grouped by URL
    effective_k = top_k * 2 if config.USE_PAGE_INDEX else top_k
    effective_fetch_k = config.RETRIEVE_K * 2 if config.USE_PAGE_INDEX else config.RETRIEVE_K

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": effective_k,
            "fetch_k": effective_fetch_k,
            "lambda_mult": 0.7,
        },
    )

    # Create LLM
    llm = create_llm(model_name=model_name, temperature=temperature)

    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{question}"),
        ]
    )

    # Retrieve documents
    docs = retriever.invoke(question)
    context = _format_docs(docs)

    # Convert chat history to messages
    messages = []
    if chat_history:
        for human_msg, ai_msg in chat_history:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))

    # Build chain
    chain = prompt | llm | StrOutputParser()

    invoke_args = {
        "context": context,
        "question": question,
        "chat_history": messages,
    }

    # Stream with rate limiting and retry
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            _rate_limit_wait()
            yield from chain.stream(invoke_args)
            return  # Success, exit
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "429" in error_msg or "limit" in error_msg:
                wait_time = config.LLM_MIN_REQUEST_INTERVAL * (2 ** (attempt + 1))
                if attempt < config.LLM_MAX_RETRIES - 1:
                    yield f"\n\n⏳ Rate limited, retrying in {int(wait_time)}s..."
                    time.sleep(wait_time)
                    continue
                else:
                    yield "\n\n⚠️ Rate limit exceeded. Please wait a minute and try again."
                    return
            # Non-rate-limit errors
            yield f"\n\n⚠️ Error: {e}"
            return


def interactive_chat(
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
    show_sources: bool = False,
) -> None:
    """Start an interactive chat session with the RAG system.

    Args:
        collection_name: ChromaDB collection name
        persist_directory: Path to ChromaDB persistence directory
        show_sources: Whether to display source documents for each response
    """
    console.print(
        Panel.fit(
            "[bold blue]PingIdentity Documentation Assistant[/bold blue]\n"
            "[dim]Ask questions about PingIdentity products and documentation.\n"
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

            # Query the RAG system
            console.print("[dim]Thinking...[/dim]")
            result = query(
                question=user_input,
                chat_history=chat_history,
                collection_name=collection_name,
                persist_directory=persist_directory,
                show_sources=show_sources,
            )

            # Display answer
            console.print()
            console.print("[bold cyan]Assistant:[/bold cyan]")
            console.print(Markdown(result["answer"]))

            # Display sources if enabled
            if show_sources and result.get("context"):
                console.print()
                console.print("[dim]Sources:[/dim]")
                seen_sources = set()
                for doc in result["context"]:
                    source = doc.metadata.get("source", "Unknown")
                    if source not in seen_sources:
                        seen_sources.add(source)
                        title = doc.metadata.get("title", "")
                        console.print(f"  [dim]• {title}[/dim]")
                        console.print(f"    [link={source}]{source}[/link]")

            # Add to chat history
            chat_history.append((user_input, result["answer"]))

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
