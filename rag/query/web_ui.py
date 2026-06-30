"""Gradio web UI for the RAG chatbot with streaming."""

from __future__ import annotations

import random
from pathlib import Path

import gradio as gr
from rich.console import Console

from .. import config
from ..ingestion.vector_store import load_parent_retriever, load_vector_store
from .rag_chain import query_stream

console = Console()

_FALLBACK_QUESTIONS = [
    "Getting started guide",
    "System requirements",
    "Installation steps",
    "Configuration options",
    "API reference",
    "Authentication setup",
    "Troubleshooting common errors",
    "How to upgrade",
    "Security best practices",
    "Integration guide",
]


def _sample_questions_from_db(
    collection_name: str, persist_directory: Path, n: int = 6
) -> list[str]:
    """Sample n page titles from ChromaDB and turn them into question prompts."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(persist_directory))
        col = client.get_collection(collection_name)
        total = col.count()
        if total == 0:
            return []
        # sample a spread across the index to get diverse titles
        sample_size = min(total, 200)
        result = col.get(limit=sample_size, include=["metadatas"])
        titles = list(
            {m.get("title", "").strip() for m in result["metadatas"] if m.get("title", "").strip()}
        )
        if not titles:
            return []
        chosen = random.sample(titles, min(n, len(titles)))
        return list(chosen)
    except Exception:
        return []


# Scroll the chatbot to the bottom whenever the "Searching docs…" placeholder
# appears — fires client-side as soon as Python yields the first update.
_OBSERVE_AND_SCROLL_JS = """
() => {
    function setup() {
        const wrap = document.querySelector('#chatbot .bubble-wrap');
        if (!wrap) { setTimeout(setup, 300); return; }
        new MutationObserver(() => {
            const msgs = wrap.querySelectorAll('.message');
            const last = msgs[msgs.length - 1];
            if (last && last.textContent.includes('Searching docs')) {
                wrap.scrollTop = wrap.scrollHeight;
            }
        }).observe(wrap, { childList: true, subtree: true, characterData: true });
    }
    setup();
}
"""


def _extract_text(content) -> str:
    """Normalise Gradio message content to a plain string.

    Gradio 4+ may pass content as a string or as a list of content blocks
    e.g. [{'type': 'text', 'text': '...'}].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _build_chat_history(history: list) -> list[tuple[str, str]]:
    """Convert Gradio message history to (user, assistant) tuples for the RAG chain.

    Skips error responses so they're never sent back to the model.
    """
    chat_history = []
    pending_user: str | None = None
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        text = _extract_text(msg.get("content", ""))
        if text.startswith("⚠️"):
            pending_user = None
            continue
        if role == "user":
            pending_user = text
        elif role == "assistant" and pending_user is not None:
            chat_history.append((pending_user, text))
            pending_user = None
    return chat_history


def create_chat_fn(collection_name: str, persist_directory: Path, show_sources: bool = False):
    """Return a streaming chat function for Gradio."""
    try:
        vector_store = load_vector_store(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    except Exception:

        def error_fn(message: str, history: list):
            yield "❌ No vector store found. Please run `python main.py ingest` first."

        return error_fn

    def chat_fn(message: str, history: list):
        if not message.strip():
            yield ""
            return

        chat_history = _build_chat_history(history)

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

        full_response = ""
        for chunk in query_stream(
            question=message,
            chat_history=chat_history or None,
            collection_name=collection_name,
            persist_directory=persist_directory,
        ):
            full_response += chunk
            yield full_response

        if show_sources and source_docs:
            sources_text = "\n\n---\n\n**📚 Sources:**\n"
            seen: set[str] = set()
            for doc in source_docs:
                source = doc.metadata.get("source", "Unknown")
                title = doc.metadata.get("title", "Untitled")
                if source not in seen:
                    seen.add(source)
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
    root_path: str = "",
) -> None:
    """Launch the Gradio web interface."""
    # Load retriever synchronously before accepting connections — no race condition.
    if config.USE_PARENT_RETRIEVER:
        load_parent_retriever(collection_name, persist_directory)
    else:
        load_vector_store(collection_name, persist_directory)

    chat_fn = create_chat_fn(collection_name, persist_directory, show_sources=show_sources)
    examples = _sample_questions_from_db(collection_name, persist_directory, n=6)
    if not examples:
        examples = random.sample(_FALLBACK_QUESTIONS, 6)
        # print("Example questions: using fallback (no DB titles found)", flush=True)
    # else:
    # print("Example questions from DB:", flush=True)
    # for q in examples:
    # print(f"  • {q}", flush=True)

    site_name = config.SITE_NAME
    with gr.Blocks(title=f"{site_name} Docs Assistant") as demo:
        demo.load(fn=None, js=_OBSERVE_AND_SCROLL_JS)

        gr.Markdown(f"# {site_name} Documentation Assistant")
        gr.Markdown(f"Ask questions about {site_name} documentation.")

        chatbot = gr.Chatbot(label="Chat", height=400, autoscroll=False, elem_id="chatbot")

        gr.Markdown("### 💡 Try these questions:")
        with gr.Row():
            example_btns = [gr.Button(q, size="sm") for q in examples[:3]]
        with gr.Row():
            example_btns += [gr.Button(q, size="sm") for q in examples[3:]]

        msg = gr.Textbox(
            label="Your Question",
            placeholder=f"Ask about {site_name} documentation...",
            lines=2,
        )
        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")

        all_outputs = [chatbot, msg, submit_btn, clear_btn] + example_btns

        def _lock():
            return (
                [gr.update(interactive=False)]
                + [gr.update(value="⏳ Generating…", interactive=False)]
                + [gr.update(interactive=False)]
                + [gr.update(interactive=False)] * 6
            )

        def _unlock():
            return (
                [gr.update(interactive=True)]
                + [gr.update(value="Send", interactive=True)]
                + [gr.update(interactive=True)]
                + [gr.update(interactive=True)] * 6
            )

        def respond(message: str, history: list):
            if not message or not message.strip():
                yield [gr.update(value=history, autoscroll=False)] + _unlock()
                return

            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "_Searching docs…_"},
            ]
            yield [gr.update(value=history, autoscroll=True)] + _lock()

            first = True
            for response in chat_fn(message, history[:-2]):
                history[-1]["content"] = response + ("▌" if not first else "")
                first = False
                yield [gr.update(value=history, autoscroll=False)] + _lock()

            history[-1]["content"] = history[-1]["content"].removesuffix("▌")
            yield [gr.update(value=history, autoscroll=False)] + _unlock()

        msg.submit(respond, [msg, chatbot], all_outputs)
        submit_btn.click(respond, [msg, chatbot], all_outputs)
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

        for btn, example_text in zip(example_btns, examples, strict=False):
            btn.click(lambda txt=example_text: txt, outputs=msg).then(
                respond, inputs=[msg, chatbot], outputs=all_outputs
            )

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        root_path=root_path,
    )
