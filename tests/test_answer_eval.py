from rxcite.answer_eval import (
    AnswerRecord,
    choose_threshold,
    judge_faithfulness,
    lines,
    make_unanswerable,
    mentions_known_drug,
    refusal_rates,
    run_one,
    split_halves,
)
from rxcite.models import Answer, Hit
from rxcite.retrieval import Retriever
from tests.fakes import FakeEmbedder, FakeIndex, ScriptedLLM, chunk


def record(kind: str, score: float, refused: bool = False, qid: str = "q") -> AnswerRecord:
    answer = Answer(
        question="?", answer="a", citations=[], refused=refused, retrieval_ms=1, generation_ms=1
    )
    return AnswerRecord(
        question_id=qid,
        kind=kind,
        question="?",
        top_score=score,
        answer=answer,
        faithful=None if refused else True,
        judge_reason="",
    )


def test_lines_strips_numbering() -> None:
    assert lines("1. How do I bake bread?\n- What is a comet?\nshort\n") == [
        "How do I bake bread?",
        "What is a comet?",
    ]


def test_mentions_known_drug() -> None:
    known = ["Ibuprofen (Advil)", "Acetaminophen, Codeine"]
    assert mentions_known_drug("Can I take ibuprofen daily?", known)
    assert not mentions_known_drug("Is felbamate safe?", known)


def test_make_unanswerable_filters_known_drugs() -> None:
    class TwoReplies:
        def __init__(self) -> None:
            self.replies = [
                "1. How do I bake bread?\n2. Who painted the Mona Lisa?",
                "Is felbamate safe?\nCan ibuprofen cause rashes?",
            ]

        def complete(self, prompt: str, system: str = "") -> str:
            return self.replies.pop(0)

    pairs = make_unanswerable(TwoReplies(), ["Ibuprofen"], ["Felbamate", "Ibuprofen"], 2, 5)
    assert pairs == [
        ("off_topic", "How do I bake bread?"),
        ("off_topic", "Who painted the Mona Lisa?"),
        ("unknown_drug", "Is felbamate safe?"),
    ]


def test_refusal_rates_and_threshold() -> None:
    recs = [
        record("answerable", 0.80),
        record("answerable", 0.72, refused=True),
        record("unanswerable", 0.50),
        record("unanswerable", 0.75, refused=True),
        record("unanswerable", 0.78),
    ]
    assert choose_threshold(recs) == 0.72
    rates = refusal_rates(recs, 0.72)
    assert rates["false_refusal_rate"] == 0.5  # the LLM refused one answerable question
    assert rates["correct_refusal_rate"] == 2 / 3
    assert rates["pre_llm_refusal_rate"] == 1 / 3
    assert refusal_rates(recs, None)["pre_llm_refusal_rate"] == 0.0


def test_split_halves_is_stratified() -> None:
    recs = [record("answerable", 0.8, qid=f"a{i}") for i in range(4)]
    recs += [record("unanswerable", 0.5, qid=f"u{i}") for i in range(4)]
    cal, test = split_halves(recs, seed=1)
    assert sorted(r.kind for r in cal) == ["answerable"] * 2 + ["unanswerable"] * 2
    assert {r.question_id for r in cal}.isdisjoint({r.question_id for r in test})


def test_run_one_judges_answered_questions() -> None:
    from judgekit.judges import LLMJudge

    hits = [Hit(chunk=chunk("a", "May cause drowsiness."), score=0.8)]
    retriever = Retriever(FakeIndex(hits, []), FakeEmbedder(), mode="vector")
    judge = LLMJudge("groundedness", ScriptedLLM('{"reason": "supported", "verdict": "pass"}'))
    rec = run_one(
        "q1",
        "answerable",
        "Does it cause drowsiness?",
        retriever,
        ScriptedLLM("Yes [1]."),
        judge,
        top_k=1,
    )
    assert rec.faithful is True and rec.top_score == 0.8
    passed, _ = judge_faithfulness(judge, "?", hits, "Yes [1].")
    assert passed
    refused = run_one(
        "q2",
        "unanswerable",
        "?",
        Retriever(FakeIndex([], []), FakeEmbedder(), mode="vector"),
        ScriptedLLM("x"),
        judge,
        top_k=1,
    )
    assert refused.faithful is None and refused.answer.refused
