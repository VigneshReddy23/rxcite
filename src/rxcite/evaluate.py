"""Measure retrieval quality on a question set with known answers.

Question set: for a seeded random sample of passages, an LLM writes one
realistic question that the passage answers (in plain words, not copying the
label's wording). That passage is the question's "gold" source.

A retrieved passage counts as relevant if it comes from the same drug label
AND the same section as the gold passage (sections are split into several
overlapping chunks, so the answer can sit in a neighbouring chunk). Exact-chunk
matching is reported too, as a stricter view.

Metrics:
  recall@k  share of questions whose relevant passage is in the top k
  MRR       mean of 1/rank of the first relevant passage (0 if not in top 10)
"""

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from judgekit.providers import Provider
from pydantic import BaseModel

from rxcite.models import Chunk, Hit
from rxcite.retrieval import Retriever

QUESTION_SYSTEM = """You write evaluation questions for a medicine search engine.
Given one passage from an FDA drug label, write ONE question that a patient,
caregiver or pharmacist might realistically ask, which this passage answers.

Rules:
- Name the drug the way a person would (a common generic or brand name).
- Use everyday words and paraphrase; do not copy phrases from the passage.
- The question must be answerable from this passage alone.
- Output only the question, nothing else."""

KS = (1, 3, 5, 10)
MIN_WORDS = 40  # skip tiny passages ("Store at 20-25C") that make trivial questions


class Question(BaseModel):
    id: str
    question: str
    gold_chunk_id: str
    set_id: str
    field: str  # label section field, e.g. "warnings"


def section_key(chunk_id: str) -> str:
    """'<set_id>:<field>:<n>' -> '<set_id>:<field>'."""
    return chunk_id.rsplit(":", 1)[0]


def sample_chunks(chunks: Sequence[Chunk], n: int, seed: int) -> list[Chunk]:
    eligible = [c for c in chunks if len(c.text.split()) >= MIN_WORDS]
    return random.Random(seed).sample(eligible, n)


def make_questions(
    chunks: Sequence[Chunk], provider: Provider, n: int, seed: int
) -> list[Question]:
    questions = []
    for i, c in enumerate(sample_chunks(chunks, n, seed), start=1):
        prompt = (
            f"<drug>{c.drug}</drug>\n<section>{c.section}</section>\n"
            f"<passage>\n{c.text}\n</passage>"
        )
        text = provider.complete(prompt, system=QUESTION_SYSTEM).strip().strip('"')
        set_id, field, _ = c.id.rsplit(":", 2)
        questions.append(
            Question(id=f"q{i:03d}", question=text, gold_chunk_id=c.id, set_id=set_id, field=field)
        )
    return questions


def save_questions(questions: Sequence[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(q.model_dump_json() + "\n" for q in questions), encoding="utf-8")


def load_questions(path: Path) -> list[Question]:
    return [Question.model_validate_json(line) for line in path.read_text().splitlines() if line]


def first_relevant_rank(hits: Sequence[Hit], q: Question, exact: bool = False) -> int | None:
    """1-based rank of the first relevant hit, or None."""
    for rank, hit in enumerate(hits, start=1):
        if exact and hit.chunk.id == q.gold_chunk_id:
            return rank
        if not exact and section_key(hit.chunk.id) == section_key(q.gold_chunk_id):
            return rank
    return None


def metrics(ranks: Sequence[int | None], ks: Sequence[int] = KS) -> dict[str, float]:
    n = len(ranks)
    out = {f"recall@{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks}
    out["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return out


def evaluate_retrieval(
    retriever: Retriever, questions: Sequence[Question], depth: int = max(KS)
) -> dict[str, Any]:
    section_ranks, exact_ranks = [], []
    for q in questions:
        hits = retriever.search(q.question, depth)
        section_ranks.append(first_relevant_rank(hits, q))
        exact_ranks.append(first_relevant_rank(hits, q, exact=True))
    return {
        "mode": retriever.mode,
        "n": len(questions),
        "section": metrics(section_ranks),
        "exact_chunk": metrics(exact_ranks),
    }


def format_table(results: Sequence[dict[str, Any]], view: str = "section") -> str:
    header = "| mode | " + " | ".join(f"recall@{k}" for k in KS) + " | MRR |"
    lines = [header, "|" + "---|" * (len(KS) + 2)]
    for r in results:
        m = r[view]
        cells = " | ".join(f"{m[f'recall@{k}']:.3f}" for k in KS)
        lines.append(f"| {r['mode']} | {cells} | {m['mrr']:.3f} |")
    return "\n".join(lines)


def save_results(results: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(results), indent=2), encoding="utf-8")
