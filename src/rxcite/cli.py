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


UNANSWERABLE = Path("eval/unanswerable.jsonl")
ANSWERS = Path("results/answers.jsonl")


@app.command("make-unanswerable")
def make_unanswerable_cmd(
    off_topic: Annotated[int, typer.Option(help="Off-topic questions.")] = 20,
    unknown_drug: Annotated[int, typer.Option(help="Questions about drugs not indexed.")] = 20,
    out: Annotated[Path, typer.Option(help="Where to save them.")] = UNANSWERABLE,
) -> None:
    """Generate questions the FDA labels in the corpus can't answer."""
    import json

    from judgekit.providers import make_provider

    from rxcite.answer_eval import make_unanswerable
    from rxcite.config import load_settings

    drugs = sorted({ingest.drug_name(label) for label in ingest.load_labels(RAW)})
    # Real drug names well below the indexed top 300 in openFDA's label counts.
    candidates = [n.title() for n in ingest.top_generic_names(1000)[600:]]
    pairs = make_unanswerable(
        make_provider(load_settings().llm), drugs, candidates, off_topic, unknown_drug
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(
            json.dumps({"id": f"u{i:03d}", "subtype": kind, "question": q}) + "\n"
            for i, (kind, q) in enumerate(pairs, start=1)
        )
    )
    typer.echo(f"Saved {len(pairs)} unanswerable questions to {out}")


@app.command("eval-answers")
def eval_answers_cmd(
    answerable: Annotated[int, typer.Option(help="Answerable questions to sample.")] = 60,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 7,
    out: Annotated[Path, typer.Option(help="Where to save per-question records.")] = ANSWERS,
) -> None:
    """Answer answerable + unanswerable questions and judge faithfulness with judgekit."""
    import json
    import random

    from judgekit.judges import LLMJudge
    from judgekit.providers import make_provider

    from rxcite import db
    from rxcite.answer_eval import run_one, save_records
    from rxcite.config import load_settings
    from rxcite.embeddings import FastEmbedder
    from rxcite.evaluate import load_questions
    from rxcite.retrieval import PostgresIndex, Retriever

    settings = load_settings()
    provider = make_provider(settings.llm)
    judge = LLMJudge("groundedness", provider)  # judgekit's calibrated rubric
    retriever = Retriever(
        PostgresIndex(db.connect(settings.database_url)), FastEmbedder(), mode="vector"
    )

    qs = random.Random(seed).sample(load_questions(QUESTIONS), answerable)
    todo = [(q.id, "answerable", q.question) for q in qs]
    todo += [
        (row["id"], "unanswerable", row["question"])
        for row in map(json.loads, UNANSWERABLE.read_text().splitlines())
    ]
    records = []
    for i, (qid, kind, question) in enumerate(todo, start=1):
        records.append(run_one(qid, kind, question, retriever, provider, judge, settings.top_k))
        if i % 20 == 0:
            typer.echo(f"  {i}/{len(todo)}")
    save_records(records, out)
    typer.echo(f"Saved {len(records)} records to {out}")


@app.command("report-answers")
def report_answers_cmd(
    records_path: Annotated[Path, typer.Option("--records", help="From eval-answers.")] = ANSWERS,
    seed: Annotated[int, typer.Option(help="Calibration/test split seed.")] = 11,
) -> None:
    """Faithfulness, and refusal rates with a threshold chosen on a held-out split."""
    from rxcite.answer_eval import choose_threshold, load_records, refusal_rates, split_halves

    records = load_records(records_path)
    answered = [r for r in records if r.kind == "answerable" and r.faithful is not None]
    faithful = sum(bool(r.faithful) for r in answered)
    typer.echo(
        f"Faithfulness (answerable, answered): {faithful}/{len(answered)} "
        f"= {faithful / len(answered):.3f}"
    )
    cited = sum(bool(r.answer.citations) for r in answered)
    typer.echo(f"Answers with at least one citation: {cited}/{len(answered)}")

    cal, test = split_halves(records, seed)
    threshold = choose_threshold(cal)
    typer.echo(f"\nThreshold chosen on calibration half: top cosine < {threshold:.3f} -> refuse")
    for name, t in [("LLM refusal only", None), ("LLM + threshold", threshold)]:
        rates = refusal_rates(test, t)
        typer.echo(
            f"  test half, {name:<17} false refusals {rates['false_refusal_rate']:.3f}"
            f" | correct refusals {rates['correct_refusal_rate']:.3f}"
            f" | refused before any LLM call {rates['pre_llm_refusal_rate']:.3f}"
        )


@app.command("report-cost")
def report_cost_cmd(
    records_path: Annotated[Path, typer.Option("--records", help="From eval-answers.")] = ANSWERS,
    input_price: Annotated[float, typer.Option(help="USD per 1M input tokens.")] = 1.10,
    output_price: Annotated[float, typer.Option(help="USD per 1M output tokens.")] = 5.50,
) -> None:
    """Latency percentiles and cost per 1,000 queries from recorded answers.

    Default prices: Claude Haiku 4.5 on Bedrock, us-east-1 "Standard" tier
    (AWS Marketplace rate card, checked 2026-09-28).
    """
    from rxcite.answer_eval import load_records
    from rxcite.bench import cost_per_1k, latency_summary

    records = load_records(records_path)
    for name, p in latency_summary(records).items():
        typer.echo(f"{name:<14} p50 {p['p50']:7.0f} ms   p95 {p['p95']:7.0f} ms")
    c = cost_per_1k(records, input_price, output_price)
    typer.echo(
        f"\nPer LLM call: {c['input_tokens']:.0f} input + {c['output_tokens']:.0f} output tokens"
        f"\nCost: ${c['usd_per_1k']:.2f} per 1,000 answered questions"
        f" (${input_price}/${output_price} per 1M tokens)"
    )
