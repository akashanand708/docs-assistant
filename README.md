# PingIdentity Documentation Assistant

A RAG (Retrieval-Augmented Generation) based AI assistant for querying PingIdentity documentation using natural language.

Uses **GitHub Models API** for LLM responses and **local embeddings** for vector search.

## Screenshot

![PingIdentity Documentation Assistant](assets/web-ui.png)

## How It Works

1. **Scrape documentation** → Crawls docs.pingidentity.com automatically
2. **Process & embed** → Chunks text and creates vector embeddings locally
3. **You ask questions** → Natural language queries about PingIdentity products
4. **RAG retrieves** → Finds relevant documentation chunks via similarity search
5. **LLM answers** → GPT-4o-mini generates accurate responses with source citations

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Question                        │
│  "How do I configure PingFederate for SAML SSO?"       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                         │
│                                                         │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │  ChromaDB       │───▶│  Top-K Similar Chunks       │ │
│  │  Vector Store   │    │  (MMR for diversity)        │ │
│  └─────────────────┘    └─────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Context + Question → GitHub Models (GPT-4o-mini)  │ │
│  └─────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Response                             │
│  Answer with source URLs and documentation references   │
└─────────────────────────────────────────────────────────┘
```

## Features

- **Web Scraper** - Crawls docs.pingidentity.com automatically
- **Local Embeddings** - No API rate limits (sentence-transformers)
- **GitHub Models** - GPT-4o-mini via GitHub Models API
- **Source Citations** - Shows documentation URLs with responses
- **Shareable Web UI** - Gradio interface with public link option
- **Rate Limit Handling** - Auto-retry with exponential backoff

## Technologies Used

| Category | Technology |
|----------|------------|
| **Language** | Python 3.10+ |
| **LLM** | GPT-4o-mini via GitHub Models API |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector Store** | ChromaDB |
| **RAG Framework** | LangChain |
| **Web UI** | Gradio |
| **CLI** | Typer |
| **Web Scraping** | BeautifulSoup, requests |

## Project Structure

```
pingid-docs-assistant/
├── main.py              # CLI entry point (8 commands)
├── requirements.txt     # Python dependencies
├── doc.md               # Detailed setup guide
├── rag/
│   ├── config.py        # Configuration settings
│   ├── scraper.py       # Web scraper
│   ├── processor.py     # Document chunking
│   ├── vector_store.py  # ChromaDB operations
│   ├── rag_chain.py     # LangChain RAG pipeline
│   └── web_ui.py        # Gradio web interface
└── chroma_db/           # Vector database (after ingest)
```

## Documentation

See [doc.md](doc.md) for:
- Installation & prerequisites
- Configuration (environment variables)
- CLI command reference
- Docker deployment
- Tuning guide
- Troubleshooting

## License

MIT
