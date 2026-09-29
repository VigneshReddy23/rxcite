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
        db.upsert_chunks(conn, part, embedder.embed([c.text for c in part]))
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
