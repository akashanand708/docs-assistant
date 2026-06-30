"""Evaluation and benchmarking for the RAG pipeline.

Two independent benchmarks:

  1. Hit Rate / MRR  — pure retrieval accuracy.
     Does NOT need an LLM.  Only needs the vector index and a test set with
     (question, expected_url_fragment) pairs.

  2. RAGAS           — generation quality evaluation.
     Needs ground-truth answers in the test set in addition to retrieval.
     Metrics: faithfulness, answer_relevancy, context_recall, context_precision.

Run from the CLI:
    python main.py evaluate                          # hit rate only (fast)
    python main.py evaluate --ragas                  # hit rate + RAGAS
    python main.py evaluate --testset my_tests.json  # custom test set
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from . import config

console = Console()

_TESTSET_GENERATION_PROMPT = """You are building an evaluation dataset for a documentation RAG system. Given a documentation page, generate {n} realistic user questions and reference answers based solely on the content below.

Rules:
- Questions must be answerable solely from the content below
- Vary question types: how-to, conceptual, troubleshooting, configuration
- Be specific, not generic ("How do I enable X?" not "How do I configure things?")
- Each question should target a different part of the content
- Answers should be concise but complete (2-4 sentences), drawn directly from the content

Page URL: {url}
Page Title: {title}
Content:
{content}

Respond with a JSON array of objects, nothing else. Example:
[{{"question": "Question 1?", "answer": "Answer 1."}}, {{"question": "Question 2?", "answer": "Answer 2."}}]"""


def generate_testset(
    collection_name: str = config.COLLECTION_NAME,  # noqa: ARG001 — kept for API compat, parent store used instead
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
    output_path: Path | str = Path("eval_testset.json"),
    n_pages: int = 20,
    questions_per_page: int = 2,
) -> Path:
    """Sample pages from ChromaDB and generate evaluation questions via LLM.

    Writes eval_testset.json with expected_url set to each page's URL so
    hit-rate evaluation works immediately without manual annotation.

    Args:
        collection_name: ChromaDB collection to sample from.
        persist_directory: Path to ChromaDB persist directory.
        output_path: Where to write the generated test set.
        n_pages: Number of distinct pages to sample.
        questions_per_page: Questions to generate per page.

    Returns:
        Path to the written test set file.
    """
    from .query.rag_chain import _build_llm

    output_path = Path(output_path)
    persist_dir = Path(persist_directory)

    console.print(
        f"\n[bold blue]── Generating eval test set ({n_pages} pages × {questions_per_page} questions) ──[/bold blue]"
    )

    # Read from parent store — full parent sections (up to PARENT_CHUNK_SIZE chars),
    # one per UUID file, much richer than the child chunks stored in ChromaDB.
    parent_store_dir = persist_dir / "parent_store"
    if not parent_store_dir.exists():
        raise FileNotFoundError(
            f"Parent store not found at {parent_store_dir}. Run 'ingest' first."
        )

    parent_files = list(parent_store_dir.iterdir())
    console.print(f"[dim]Found {len(parent_files)} parent sections in store[/dim]")

    # Parse each file and deduplicate by source URL, keeping the longest section per page
    seen: dict[str, dict] = {}
    for path in parent_files:
        try:
            with open(path) as f:
                doc = json.load(f)
            kw = doc.get("kwargs", {})
            content = kw.get("page_content", "")
            meta = kw.get("metadata", {})
            url = meta.get("source", "")
            title = meta.get("title", "")
            if not url or len(content) < 300:
                continue
            # Keep the longest section for each page — most content for question generation
            if url not in seen or len(content) > len(seen[url]["content"]):
                seen[url] = {"url": url, "title": title, "content": content}
        except Exception:
            continue

    pages = list(seen.values())
    if len(pages) > n_pages:
        pages = random.sample(pages, n_pages)

    console.print(f"[dim]Sampled {len(pages)} unique pages from parent store[/dim]")

    llm = _build_llm(config.LLM_MODEL, temperature=0.7)
    test_cases: list[dict] = []

    for i, page in enumerate(pages, 1):
        console.print(f"  [dim]({i}/{len(pages)}) {page['title'][:60] or page['url'][:60]}[/dim]")

        content_preview = page["content"][:2000]
        prompt = _TESTSET_GENERATION_PROMPT.format(
            n=questions_per_page,
            url=page["url"],
            title=page["title"],
            content=content_preview,
        )

        try:
            from langchain_core.messages import HumanMessage

            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content
            if isinstance(raw, list):
                raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)

            # Extract JSON array from response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            items: list = json.loads(raw[start:end]) if start != -1 else []

            for item in items[:questions_per_page]:
                if isinstance(item, dict):
                    question = item.get("question", "")
                    ground_truth = item.get("answer", "")
                else:
                    question = str(item)
                    ground_truth = ""
                if question:
                    test_cases.append(
                        {
                            "question": question,
                            "expected_url": page["url"],
                            "ground_truth": ground_truth,
                        }
                    )
        except Exception as e:
            console.print(f"    [yellow]Skipped (error: {e})[/yellow]")

        time.sleep(0.5)  # avoid hammering Bedrock

    random.shuffle(test_cases)

    with open(output_path, "w") as f:
        json.dump(test_cases, f, indent=2)

    console.print(
        f"[bold green]Generated {len(test_cases)} test cases → {output_path}[/bold green]"
    )
    return output_path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """A single evaluation test case."""

    question: str
    # Substring that must appear in the URL of at least one retrieved doc.
    # e.g. "oidc"  matches  "https://docs.example.com/r/en-US/oidc-config"
    expected_url: str = ""
    # Optional: reference answer used by RAGAS metrics.
    ground_truth: str = ""


@dataclass
class HitRateResult:
    """Retrieval benchmark results."""

    k: int
    total: int
    hits: int
    mrr_sum: float
    per_case: list[dict[str, Any]] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def mrr(self) -> float:
        """Mean Reciprocal Rank — rewards hitting the right doc higher in the list."""
        return self.mrr_sum / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# Test-set loading
# ---------------------------------------------------------------------------


def load_testset(path: Path) -> list[TestCase]:
    """Load test cases from a JSON file.

    Expected format:
    [
      {
        "question": "How do I configure OIDC?",
        "expected_url": "oidc",
        "ground_truth": "To configure OIDC..."   // optional, needed for RAGAS
      },
      ...
    ]
    """
    with open(path) as f:
        data = json.load(f)

    cases = []
    for item in data:
        cases.append(
            TestCase(
                question=item["question"],
                expected_url=item.get("expected_url", ""),
                ground_truth=item.get("ground_truth", ""),
            )
        )
    console.print(f"[dim]Loaded {len(cases)} test cases from {path}[/dim]")
    return cases


# ---------------------------------------------------------------------------
# Benchmark 1: Hit Rate + MRR
# ---------------------------------------------------------------------------


def run_hit_rate(
    cases: list[TestCase],
    k: int = config.TOP_K_RESULTS,
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
) -> HitRateResult:
    """Measure retrieval accuracy: Hit Rate@k and MRR.

    For each test case, retrieves top-k documents and checks whether any
    retrieved document's URL contains the expected_url substring.

    Hit Rate = fraction of questions where the right page was in top-k.
    MRR      = mean of 1/rank of the first correct hit (rewards higher placement).

    Args:
        cases: Test cases with expected_url set.
        k: Number of documents to retrieve per question.
        collection_name: ChromaDB collection name.
        persist_directory: Path to ChromaDB persist directory.

    Returns:
        HitRateResult with detailed per-case breakdown.
    """
    # Only evaluate cases that have an expected_url
    eval_cases = [c for c in cases if c.expected_url]
    if not eval_cases:
        console.print("[yellow]No test cases have expected_url — skipping hit rate.[/yellow]")
        return HitRateResult(k=k, total=0, hits=0, mrr_sum=0.0)

    console.print(f"\n[bold blue]── Retrieval Benchmark (Hit Rate @{k}) ──[/bold blue]")
    console.print(f"[dim]Evaluating {len(eval_cases)} questions...[/dim]\n")

    if config.USE_PARENT_RETRIEVER:
        from .ingestion.vector_store import load_parent_retriever

        retriever = load_parent_retriever(collection_name, persist_directory)
        retriever.search_kwargs = {"k": k * 2}  # Wide child search
    else:
        from .ingestion.vector_store import load_vector_store

        vs = load_vector_store(collection_name, persist_directory)
        retriever = vs.as_retriever(search_kwargs={"k": k})

    hits = 0
    mrr_sum = 0.0
    per_case = []

    for case in eval_cases:
        docs = retriever.invoke(case.question)
        retrieved_urls = [d.metadata.get("source", "") for d in docs]

        # Find the first rank that matches
        hit_rank: int | None = None
        for rank, url in enumerate(retrieved_urls, start=1):
            if case.expected_url.lower() in url.lower():
                hit_rank = rank
                break

        is_hit = hit_rank is not None
        if is_hit:
            hits += 1
            mrr_sum += 1.0 / hit_rank

        status_icon = "[green]✓[/green]" if is_hit else "[red]✗[/red]"
        rank_str = f"rank {hit_rank}" if is_hit else "not found"
        console.print(f"  {status_icon}  {case.question[:70]:<70}  [{rank_str}]")

        per_case.append(
            {
                "question": case.question,
                "expected_url": case.expected_url,
                "hit": is_hit,
                "hit_rank": hit_rank,
                "retrieved_urls": retrieved_urls[:k],
            }
        )

    return HitRateResult(k=k, total=len(eval_cases), hits=hits, mrr_sum=mrr_sum, per_case=per_case)


# ---------------------------------------------------------------------------
# Benchmark 2: RAGAS
# ---------------------------------------------------------------------------


def run_ragas(
    cases: list[TestCase],
    k: int = config.TOP_K_RESULTS,
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path | str = config.CHROMA_PERSIST_DIR,
) -> dict[str, float]:
    """Run RAGAS evaluation metrics.

    Metrics computed:
      - context_precision:  Are the retrieved docs relevant to the question?
      - context_recall:     Did we retrieve all the info needed to answer?
      - faithfulness:       Does the answer stay within the retrieved context?
      - answer_relevancy:   Does the answer actually address the question?

    Ground truth answers are required in the test cases.

    Args:
        cases: Test cases; must have ground_truth set for RAGAS.
        k: Number of documents to retrieve per question.
        collection_name: ChromaDB collection name.
        persist_directory: Path to ChromaDB persist directory.

    Returns:
        Dict of metric_name → score (0.0–1.0).
    """
    try:
        from ragas import evaluate  # type: ignore
        from ragas.dataset_schema import SingleTurnSample  # type: ignore
        from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
        from ragas.llms import LangchainLLMWrapper  # type: ignore
        from ragas.metrics import (  # type: ignore
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as e:
        console.print(f"[red]Failed to import RAGAS: {e}[/red]")
        console.print("[dim]Try: pip install ragas==0.2.15[/dim]")
        return {}

    _THIN_MARKERS = (
        "no further",
        "not included",
        "not provided",
        "not available",
        "only the title",
        "content provided does not",
        "no content",
        "no details",
    )

    def _is_useful_ground_truth(gt: str) -> bool:
        gt_lower = gt.lower()
        return bool(gt) and not any(m in gt_lower for m in _THIN_MARKERS)

    # Only evaluate cases that have substantive ground truth
    eval_cases = [c for c in cases if _is_useful_ground_truth(c.ground_truth)]
    if not eval_cases:
        console.print(
            "[yellow]No test cases have ground_truth — skipping RAGAS.[/yellow]\n"
            "[dim]Add a 'ground_truth' field to your test cases to enable RAGAS.[/dim]"
        )
        return {}

    console.print("\n[bold blue]── RAGAS Benchmark ──[/bold blue]")
    console.print(f"[dim]Evaluating {len(eval_cases)} questions with ground truth...[/dim]\n")

    # Pop AWS_PROFILE for the entire block — pydantic re-validates credentials
    # on every ChatBedrockConverse instantiation inside RAGAS internals.
    import os

    saved_profile = os.environ.pop("AWS_PROFILE", None)
    try:
        # Set up retriever
        if config.USE_PARENT_RETRIEVER:
            from .ingestion.vector_store import load_parent_retriever

            retriever = load_parent_retriever(collection_name, persist_directory)
            retriever.search_kwargs = {"k": k * 2}
        else:
            from .ingestion.vector_store import load_vector_store

            vs = load_vector_store(collection_name, persist_directory)
            retriever = vs.as_retriever(search_kwargs={"k": k})

        # Set up LLM + embeddings
        import boto3

        region = os.environ.get("AWS_REGION", "us-east-1")
        boto_client = boto3.Session(region_name=region).client(
            "bedrock-runtime", region_name=region
        )
        from langchain_aws import ChatBedrockConverse

        from . import config as _cfg
        from .ingestion.vector_store import get_embeddings

        llm = LangchainLLMWrapper(
            ChatBedrockConverse(client=boto_client, model=_cfg.LLM_MODEL, temperature=0.1)
        )
        embeddings = LangchainEmbeddingsWrapper(get_embeddings())

        # Collect answers + contexts
        console.print("[dim]Generating answers and retrieving contexts...[/dim]")
        from .query.rag_chain import query as rag_query

        samples = []
        for case in eval_cases:
            console.print(f"  [dim]→ {case.question[:70]}[/dim]")
            docs = retriever.invoke(case.question)
            contexts = [d.page_content for d in docs]
            result = rag_query(
                question=case.question,
                collection_name=collection_name,
                persist_directory=persist_directory,
            )
            samples.append(
                SingleTurnSample(
                    user_input=case.question,
                    response=result["answer"],
                    retrieved_contexts=contexts,
                    reference=case.ground_truth,
                )
            )
            time.sleep(1.0)

        # Run RAGAS
        from ragas import EvaluationDataset  # type: ignore

        dataset = EvaluationDataset(samples=samples)
        metrics = [ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy()]

        console.print("\n[dim]Running RAGAS evaluation (this uses the LLM as judge)...[/dim]")
        ragas_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

        raw = ragas_result._scores_dict
        scores: dict[str, float] = {}
        for metric in metrics:
            name = metric.name
            try:
                vals = [v for v in raw[name] if v is not None and v == v]
                scores[name] = sum(vals) / len(vals) if vals else float("nan")
            except (KeyError, TypeError):
                scores[name] = float("nan")

        return scores

    finally:
        if saved_profile is not None:
            os.environ["AWS_PROFILE"] = saved_profile


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------


def print_hit_rate_table(result: HitRateResult) -> None:
    """Print a formatted summary table for hit rate results."""

    table = Table(title="Retrieval Benchmark Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Interpretation")

    hr = result.hit_rate
    mrr = result.mrr

    hr_interp = (
        "[red]Poor — check embeddings / chunk size[/red]"
        if hr < 0.6
        else "[yellow]Acceptable[/yellow]"
        if hr < 0.8
        else "[green]Good[/green]"
        if hr < 0.9
        else "[bold green]Excellent[/bold green]"
    )
    mrr_interp = (
        "[red]Poor[/red]"
        if mrr < 0.4
        else "[yellow]Acceptable[/yellow]"
        if mrr < 0.6
        else "[green]Good[/green]"
        if mrr < 0.8
        else "[bold green]Excellent[/bold green]"
    )

    table.add_row(f"Hit Rate @{result.k}", f"{hr:.1%}", hr_interp)
    table.add_row("MRR", f"{mrr:.3f}", mrr_interp)
    table.add_row("Questions evaluated", str(result.total), "")
    table.add_row("Hits", str(result.hits), "")

    console.print()
    console.print(table)


def print_ragas_table(scores: dict[str, float]) -> None:
    """Print a formatted summary table for RAGAS results."""
    if not scores:
        return

    table = Table(title="RAGAS Quality Benchmark", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("What it measures")
    table.add_column("Target")

    descriptions = {
        "context_precision": "Retrieved docs are relevant to the question",
        "context_recall": "All needed info was retrieved",
        "faithfulness": "Answer stays within retrieved context (no hallucination)",
        "answer_relevancy": "Answer addresses the question",
    }
    targets = {
        "context_precision": "> 0.80",
        "context_recall": "> 0.75",
        "faithfulness": "> 0.85",
        "answer_relevancy": "> 0.80",
    }

    for name, score in scores.items():
        if score != score:  # NaN
            score_str = "[dim]n/a[/dim]"
            color = "dim"
        else:
            score_str = f"{score:.3f}"
            threshold = float(targets.get(name, "> 0.75").replace("> ", ""))
            color = (
                "green" if score >= threshold else "yellow" if score >= threshold - 0.1 else "red"
            )
            score_str = f"[{color}]{score_str}[/{color}]"

        table.add_row(
            name,
            score_str,
            descriptions.get(name, ""),
            targets.get(name, ""),
        )

    console.print()
    console.print(table)


def save_results(
    hit_rate: HitRateResult | None,
    ragas_scores: dict[str, float],
    output_path: Path,
) -> None:
    """Save all benchmark results to a JSON file."""
    data: dict[str, Any] = {}

    if hit_rate:
        data["hit_rate"] = {
            "k": hit_rate.k,
            "total": hit_rate.total,
            "hits": hit_rate.hits,
            "hit_rate": round(hit_rate.hit_rate, 4),
            "mrr": round(hit_rate.mrr, 4),
            "per_case": hit_rate.per_case,
        }

    if ragas_scores:
        data["ragas"] = {k: round(v, 4) if v == v else None for k, v in ragas_scores.items()}

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"\n[dim]Results saved to {output_path}[/dim]")
