"""Command-line interface.

rxcite download --drugs 300   fetch labels from openFDA into data/raw/labels.jsonl
rxcite index                  chunk + embed + load them into Postgres
rxcite ask "question"         answer from the terminal (same pipeline as the API)
"""

from pathlib import Path
from typing import Annotated

import typer

from rxcite import ingest

app = typer.Typer(help="rxcite: cited answers from FDA drug labels.", no_args_is_help=True)

RAW = Path("data/raw/labels.jsonl")


@app.callback()
def main() -> None:
    """rxcite command-line interface."""


@app.command()
def download(
    drugs: Annotated[int, typer.Option(help="How many common drugs to fetch.")] = 300,
    out: Annotated[Path, typer.Option(help="Where to save the labels.")] = RAW,
) -> None:
    """Download one current FDA label per drug from openFDA (public domain)."""
    saved = ingest.download_labels(drugs, out)
    typer.echo(f"Saved {saved} labels to {out}")


@app.command()
def index(
    labels: Annotated[Path, typer.Option(help="Labels JSONL from `download`.")] = RAW,
    batch: Annotated[int, typer.Option(help="Chunks embedded per batch.")] = 256,
) -> None:
    """Chunk the labels, embed each chunk locally, and load them into Postgres."""
    from rxcite import db
    from rxcite.config import load_settings
    from rxcite.embeddings import FastEmbedder

    chunks = [c for label in ingest.load_labels(labels) for c in ingest.chunk_label(label)]
    typer.echo(f"{len(chunks)} chunks from {labels}")
    conn = db.connect(load_settings().database_url)
    db.init_schema(conn)
    embedder = FastEmbedder()
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        db.upsert_chunks(conn, part, embedder.embed([c.with_context() for c in part]))
        typer.echo(f"  indexed {min(start + batch, len(chunks))}/{len(chunks)}")
    typer.echo(f"Database now holds {db.count_chunks(conn)} chunks")


@app.command()
def ask(question: Annotated[str, typer.Argument(help="Your question.")]) -> None:
    """Answer one question with citations."""
    from rxcite.api import build_service

    result = build_service().ask(question)
    typer.echo(result.answer)
    for c in result.citations:
        typer.echo(f"  [{c.number}] {c.drug}, {c.section}: {c.url}")
    typer.echo(
        f"(retrieval {result.retrieval_ms:.0f} ms, generation {result.generation_ms:.0f} ms)"
    )


QUESTIONS = Path("eval/questions.jsonl")


@app.command("make-questions")
def make_questions_cmd(
    n: Annotated[int, typer.Option(help="How many questions.")] = 150,
    seed: Annotated[int, typer.Option(help="Random seed (recorded).")] = 42,
    out: Annotated[Path, typer.Option(help="Where to save them.")] = QUESTIONS,
) -> None:
    """Generate the retrieval test set: one LLM-written question per sampled passage."""
    from judgekit.providers import make_provider

    from rxcite.config import load_settings
    from rxcite.evaluate import make_questions, save_questions

    chunks = [c for label in ingest.load_labels(RAW) for c in ingest.chunk_label(label)]
    provider = make_provider(load_settings().llm)
    questions = make_questions(chunks, provider, n, seed)
    save_questions(questions, out)
    typer.echo(f"Saved {len(questions)} questions to {out} (seed {seed})")


@app.command("eval-retrieval")
def eval_retrieval_cmd(
    questions: Annotated[Path, typer.Option(help="Question set.")] = QUESTIONS,
    out: Annotated[Path, typer.Option(help="Where to save results.")] = Path(
        "results/retrieval.json"
    ),
    modes: Annotated[str, typer.Option(help="Comma-separated modes, or 'all'.")] = "all",
    reranker_model: Annotated[str, typer.Option("--reranker", help="Cross-encoder model.")] = "",
) -> None:
    """Score retrieval modes on the question set (recall@k, MRR)."""
    from rxcite import db
    from rxcite.config import load_settings
    from rxcite.embeddings import RERANK_MODEL, CrossEncoderReranker, FastEmbedder
    from rxcite.evaluate import evaluate_retrieval, format_table, load_questions, save_results
    from rxcite.retrieval import MODES, PostgresIndex, Retriever

    selected = MODES if modes == "all" else tuple(m for m in MODES if m in modes.split(","))
    qs = load_questions(questions)
    conn = db.connect(load_settings().database_url)
    embedder = FastEmbedder()
    reranker = CrossEncoderReranker(reranker_model or RERANK_MODEL)
    results = []
    for mode in selected:
        retriever = Retriever(PostgresIndex(conn), embedder, reranker, mode)
        results.append(evaluate_retrieval(retriever, qs))
        typer.echo(f"  scored {mode}")
    save_results(results, out)
    typer.echo(f"\nSection-level relevance (n={len(qs)}):\n{format_table(results)}")
    typer.echo(f"\nExact-chunk relevance:\n{format_table(results, 'exact_chunk')}")
