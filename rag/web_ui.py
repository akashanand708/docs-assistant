"""Gradio web UI for the RAG chatbot with streaming."""

from __future__ import annotations

import random
from collections.abc import Generator
from pathlib import Path

import gradio as gr

from rag import config
from rag.rag_chain import query_stream
from rag.vector_store import load_vector_store

# Pool of example questions
EXAMPLE_QUESTIONS = [
    "What is PingFederate?",
    "How to configure SAML SSO?",
    "PingID MFA setup guide",
    "System requirements",
    "Authentication policies",
    "OAuth 2.0 configuration",
    "What is PingOne?",
    "How to set up LDAP?",
    "Configure OIDC integration",
    "Password reset flow",
    "User provisioning setup",
    "Access token configuration",
    "Single sign-on best practices",
    "Multi-factor authentication options",
    "API security guidelines",
    "Identity federation setup",
]


def get_random_examples(n: int = 6) -> list[str]:
    """Get n random example questions."""
    return random.sample(EXAMPLE_QUESTIONS, min(n, len(EXAMPLE_QUESTIONS)))


def create_chat_fn(collection_name: str, persist_directory: Path, show_sources: bool = False):
    """Create a streaming chat function for Gradio."""
    try:
        vector_store = load_vector_store(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    except Exception:

        def error_fn(message: str, history: list) -> Generator[str, None, None]:
            yield "❌ No vector store found. Please run `python main.py ingest` first."

        return error_fn

    def chat_fn(message: str, history: list) -> Generator[str, None, None]:
        """Stream a chat response."""
        if not message.strip():
            yield ""
            return

        # Convert messages format to chat_history tuples for RAG query
        chat_history = []
        for i in range(0, len(history) - 1, 2):
            if i + 1 < len(history):
                user_msg = history[i].get("content", "") if isinstance(history[i], dict) else ""
                ai_msg = (
                    history[i + 1].get("content", "") if isinstance(history[i + 1], dict) else ""
                )
                chat_history.append((user_msg, ai_msg))

        # Get sources first if needed (uses vector store directly, no LLM call)
        source_docs = []
        if show_sources:
            try:
                retriever = vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": config.TOP_K_RESULTS,
                        "fetch_k": config.RETRIEVE_K,
                        "lambda_mult": 0.7,
                    },
                )
                source_docs = retriever.invoke(message)
            except Exception:
                pass

        # Stream the response
        full_response = ""
        for chunk in query_stream(
            question=message,
            chat_history=chat_history if chat_history else None,
            collection_name=collection_name,
            persist_directory=persist_directory,
        ):
            full_response += chunk
            yield full_response

        # Append sources if enabled
        if show_sources and source_docs:
            sources_text = "\n\n---\n\n**📚 Sources:**\n"
            seen_sources = set()
            for doc in source_docs:
                source = doc.metadata.get("source", "Unknown")
                title = doc.metadata.get("title", "Untitled")
                if source not in seen_sources:
                    seen_sources.add(source)
                    sources_text += f"- [{title}]({source})\n"
            full_response += sources_text
            yield full_response

    return chat_fn


def launch_web_ui(
    collection_name: str,
    persist_directory: Path,
    share: bool = False,
    server_port: int = 7860,
    server_name: str = "0.0.0.0",
    show_sources: bool = False,
) -> None:
    """Launch the Gradio web interface."""
    chat_fn = create_chat_fn(collection_name, persist_directory, show_sources=show_sources)
    examples = get_random_examples(6)

    with gr.Blocks(title="PingID Docs Assistant") as demo:
        gr.Markdown("# 🔐 PingIdentity Documentation Assistant")
        gr.Markdown("Ask questions about PingIdentity products and documentation.")

        chatbot = gr.Chatbot(
            label="Chat",
            height=400,
            autoscroll=False,  # Don't force scroll while user is reading
        )

        # Example questions (randomized)
        gr.Markdown("### 💡 Try these questions:")
        with gr.Row():
            ex1 = gr.Button(examples[0], size="sm")
            ex2 = gr.Button(examples[1], size="sm")
            ex3 = gr.Button(examples[2], size="sm")
        with gr.Row():
            ex4 = gr.Button(examples[3], size="sm")
            ex5 = gr.Button(examples[4], size="sm")
            ex6 = gr.Button(examples[5], size="sm")

        msg = gr.Textbox(
            label="Your Question",
            placeholder="Ask about PingIdentity documentation...",
            lines=2,
        )

        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")

        def respond(message: str, history: list):
            """Handle user message with streaming."""
            if not message.strip():
                yield "", history
                return

            # Add user message immediately
            history = history + [{"role": "user", "content": message}]
            history = history + [{"role": "assistant", "content": ""}]

            # Stream the response
            for response in chat_fn(message, history[:-2]):  # Exclude the new messages
                history[-1]["content"] = response
                yield "", history

        def clear_chat() -> tuple[str, list]:
            """Clear chat history."""
            return "", []

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
        clear_btn.click(clear_chat, outputs=[msg, chatbot])

        # Connect example buttons - yield from respond generator
        def use_ex1(history):
            yield from respond(examples[0], history)

        def use_ex2(history):
            yield from respond(examples[1], history)

        def use_ex3(history):
            yield from respond(examples[2], history)

        def use_ex4(history):
            yield from respond(examples[3], history)

        def use_ex5(history):
            yield from respond(examples[4], history)

        def use_ex6(history):
            yield from respond(examples[5], history)

        ex1.click(use_ex1, [chatbot], [msg, chatbot])
        ex2.click(use_ex2, [chatbot], [msg, chatbot])
        ex3.click(use_ex3, [chatbot], [msg, chatbot])
        ex4.click(use_ex4, [chatbot], [msg, chatbot])
        ex5.click(use_ex5, [chatbot], [msg, chatbot])
        ex6.click(use_ex6, [chatbot], [msg, chatbot])

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
