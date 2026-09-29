from pathlib import Path

import pytest

from rxcite.evaluate import (
    Question,
    evaluate_retrieval,
    first_relevant_rank,
    format_table,
    load_questions,
    make_questions,
    metrics,
    sample_chunks,
    save_questions,
    save_results,
    section_key,
)
from rxcite.models import Hit
from rxcite.retrieval import Retriever
from tests.fakes import FakeEmbedder, FakeIndex, ScriptedLLM, chunk

LONG = " ".join(["word"] * 50)
Q = Question(
    id="q1", question="?", gold_chunk_id="set1:warnings:1", set_id="set1", field="warnings"
)


def hit(cid: str) -> Hit:
    return Hit(chunk=chunk(cid, "t"), score=1.0)


def test_section_key() -> None:
    assert section_key("abc:warnings:3") == "abc:warnings"


def test_first_relevant_rank_section_vs_exact() -> None:
    hits = [hit("set2:warnings:1"), hit("set1:warnings:0"), hit("set1:warnings:1")]
    assert first_relevant_rank(hits, Q) == 2  # neighbouring chunk of the same section counts
    assert first_relevant_rank(hits, Q, exact=True) == 3
    assert first_relevant_rank([hit("set9:dosage:0")], Q) is None


def test_metrics_hand_calculated() -> None:
    # ranks 1, 3 and a miss: recall@1 = 1/3, recall@3 = 2/3, MRR = (1 + 1/3 + 0) / 3
    m = metrics([1, 3, None], ks=(1, 3))
    assert m["recall@1"] == pytest.approx(1 / 3)
    assert m["recall@3"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 1 / 3) / 3)


def test_sample_chunks_skips_short_and_is_seeded() -> None:
    chunks = [chunk(f"s:f:{i}", LONG) for i in range(10)] + [chunk("s:f:short", "too short")]
    first = sample_chunks(chunks, 5, seed=7)
    assert first == sample_chunks(chunks, 5, seed=7)
    assert all(c.id != "s:f:short" for c in first)


def test_make_questions(tmp_path: Path) -> None:
    llm = ScriptedLLM('"Is it safe with alcohol?"')
    qs = make_questions([chunk("set1:warnings:0", LONG)], llm, n=1, seed=1)
    assert qs[0].question == "Is it safe with alcohol?"  # quotes stripped
    assert (qs[0].set_id, qs[0].field, qs[0].gold_chunk_id) == (
        "set1",
        "warnings",
        "set1:warnings:0",
    )
    assert "<passage>" in llm.calls[0][0]
    path = tmp_path / "eval" / "q.jsonl"
    save_questions(qs, path)
    assert load_questions(path) == qs


def test_evaluate_retrieval_and_table(tmp_path: Path) -> None:
    index = FakeIndex([hit("set2:x:0"), hit("set1:warnings:1")], [])
    result = evaluate_retrieval(Retriever(index, FakeEmbedder(), mode="vector"), [Q])
    assert result["mode"] == "vector"
    assert result["section"]["mrr"] == pytest.approx(0.5)
    assert result["exact_chunk"]["recall@1"] == 0.0
    table = format_table([result])
    assert table.splitlines()[0].startswith("| mode | recall@1")
    assert "| vector | 0.000 | 1.000 |" in table
    save_results([result], tmp_path / "r.json")
    assert (tmp_path / "r.json").read_text().startswith("[")
