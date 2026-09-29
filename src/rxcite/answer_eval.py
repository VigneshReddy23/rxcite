"""End-to-end answer evaluation: faithfulness and refusal behaviour.

Faithfulness is judged by judgekit's groundedness judge (calibrated against
human labels in the judgekit project) with the retrieved passages as context:
an answer passes only if every claim is supported by what was retrieved.

Refusal uses two question sets:
  answerable    questions from the retrieval test set (the labels answer them)
  unanswerable  off-topic questions + questions about drugs not in the corpus

The retrieval-confidence threshold (top cosine similarity) is chosen on one
half of each set and reported on the other half, so the reported refusal
rates aren't tuned on the questions they're measured on.
"""

import random
import re
import time
from collections.abc import Sequence
from pathlib import Path

from judgekit.judges import LLMJudge
from judgekit.models import EvalCase
from judgekit.providers import Provider
from pydantic import BaseModel

from rxcite.answer import build_prompt, generate_answer
from rxcite.models import Answer, Hit
from rxcite.retrieval import Retriever

UNANSWERABLE_SYSTEM = """Write evaluation questions for a medicine Q&A system that only
knows FDA drug labels for a fixed list of drugs. Output one question per line, nothing else."""


class AnswerRecord(BaseModel):
    question_id: str
    kind: str  # "answerable" or "unanswerable"
    question: str
    top_score: float
    answer: Answer
    faithful: bool | None  # None when refused (nothing to judge)
    judge_reason: str


def off_topic_prompt(n: int) -> str:
    return (
        f"Write {n} everyday questions that have nothing to do with medicines "
        "(sports, cooking, travel, history, technology). Vary the wording."
    )


def unknown_drug_prompt(drugs: Sequence[str]) -> str:
    names = "\n".join(drugs)
    return (
        "For EACH medicine below, write one realistic patient question about it (side effects, "
        "dosing, interactions or pregnancy), naming the medicine. One question per line, same "
        f"order, {len(drugs)} lines total.\n\n{names}"
    )


def lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip()
        if len(line) > 10:
            out.append(line)
    return out


def mentions_known_drug(question: str, known_drugs: Sequence[str]) -> bool:
    """True if any corpus drug's first word appears in the question (conservative filter)."""
    q = question.lower()
    return any(
        len(w) > 4 and re.search(rf"\b{re.escape(w)}\b", q)
        for w in {d.lower().split()[0].strip(",") for d in known_drugs}
    )


def make_unanswerable(
    provider: Provider,
    known_drugs: Sequence[str],
    candidate_drugs: Sequence[str],
    n_off_topic: int,
    n_unknown_drug: int,
    seed: int = 5,
) -> list[tuple[str, str]]:
    """(subtype, question) pairs.

    Unknown-drug questions are written about real drugs that are NOT indexed
    (candidate_drugs, e.g. less common openFDA names); any question that still
    mentions an indexed drug is dropped.
    """
    off = lines(provider.complete(off_topic_prompt(n_off_topic), system=UNANSWERABLE_SYSTEM))
    fresh = [d for d in candidate_drugs if not mentions_known_drug(d, known_drugs)]
    picked = random.Random(seed).sample(fresh, min(len(fresh), n_unknown_drug + 5))
    drug_qs = lines(provider.complete(unknown_drug_prompt(picked), system=UNANSWERABLE_SYSTEM))
    unknown = [q for q in drug_qs if not mentions_known_drug(q, known_drugs)]
    return [("off_topic", q) for q in off[:n_off_topic]] + [
        ("unknown_drug", q) for q in unknown[:n_unknown_drug]
    ]


def judge_faithfulness(
    judge: LLMJudge, question: str, hits: Sequence[Hit], answer: str
) -> tuple[bool, str]:
    context = build_prompt(question, list(hits)).split("</sources>")[0].removeprefix("<sources>\n")
    result = judge.judge(EvalCase(id="x", input=question, context=context), answer)
    return result.passed, result.reason


def run_one(
    qid: str,
    kind: str,
    question: str,
    retriever: Retriever,
    provider: Provider,
    judge: LLMJudge,
    top_k: int,
) -> AnswerRecord:
    """Answer WITHOUT a confidence threshold (so thresholds can be studied offline)."""
    start = time.perf_counter()
    hits = retriever.search(question, top_k)
    retrieval_ms = (time.perf_counter() - start) * 1000
    answer = generate_answer(question, hits, provider, retrieval_ms, refuse_below=None)
    faithful, reason = (None, "refused")
    if not answer.refused:
        faithful, reason = judge_faithfulness(judge, question, hits, answer.answer)
    return AnswerRecord(
        question_id=qid,
        kind=kind,
        question=question,
        top_score=max((h.score for h in hits), default=0.0),
        answer=answer,
        faithful=faithful,
        judge_reason=reason,
    )


def split_halves(
    records: Sequence[AnswerRecord], seed: int
) -> tuple[list[AnswerRecord], list[AnswerRecord]]:
    """Stratified 50/50 split (by kind) into calibration and test halves."""
    rng = random.Random(seed)
    cal: list[AnswerRecord] = []
    test: list[AnswerRecord] = []
    for kind in ("answerable", "unanswerable"):
        group = [r for r in records if r.kind == kind]
        rng.shuffle(group)
        cal += group[: len(group) // 2]
        test += group[len(group) // 2 :]
    return cal, test


def refusal_rates(records: Sequence[AnswerRecord], threshold: float | None) -> dict[str, float]:
    """Refusal rates when a top-score threshold is combined with the LLM's own refusals.

    pre_llm_refusal_rate: unanswerable questions refused by the threshold alone,
    i.e. before any LLM call (safer, and free).
    """

    def below(r: AnswerRecord) -> bool:
        return threshold is not None and r.top_score < threshold

    def refused(r: AnswerRecord) -> bool:
        return r.answer.refused or below(r)

    ans = [r for r in records if r.kind == "answerable"]
    una = [r for r in records if r.kind == "unanswerable"]
    return {
        "false_refusal_rate": sum(map(refused, ans)) / len(ans) if ans else 0.0,
        "correct_refusal_rate": sum(map(refused, una)) / len(una) if una else 0.0,
        "pre_llm_refusal_rate": sum(map(below, una)) / len(una) if una else 0.0,
    }


def choose_threshold(records: Sequence[AnswerRecord]) -> float:
    """The lowest top score of any answerable question: refuse only below it.

    On the calibration half this adds zero false refusals by construction; the
    test half shows whether that holds on questions it wasn't chosen on. (The
    LLM's own refusals are separate and not caused by the threshold.)
    """
    return min(r.top_score for r in records if r.kind == "answerable")


def save_records(records: Sequence[AnswerRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")


def load_records(path: Path) -> list[AnswerRecord]:
    return [AnswerRecord.model_validate_json(x) for x in path.read_text().splitlines() if x]
