import pytest

from rxcite.models import Hit
from rxcite.retrieval import CANDIDATES, Retriever, reciprocal_rank_fusion
from tests.fakes import FakeEmbedder, FakeIndex, KeywordReranker, chunk

A = chunk("a", "ibuprofen can cause stomach bleeding")
B = chunk("b", "store at room temperature")
C = chunk("c", "stop use if stomach bleeding occurs")


def hits(*chunks: object, score: float = 1.0) -> list[Hit]:
    return [Hit(chunk=c, score=score) for c in chunks]


def test_rrf_rewards_agreement_between_lists() -> None:
    fused = reciprocal_rank_fusion(hits(A, B), hits(C, A))
    assert [h.chunk.id for h in fused] == ["a", "c", "b"]  # A is in both lists
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1].score == pytest.approx(1 / 61)


def test_vector_mode() -> None:
    index = FakeIndex(hits(B, A), hits(C))
    result = Retriever(index, FakeEmbedder(), mode="vector").search("q", k=1)
    assert [h.chunk.id for h in result] == ["b"]
    assert index.calls == [("vector", CANDIDATES)]


def test_keyword_mode() -> None:
    index = FakeIndex(hits(B), hits(C, A))
    result = Retriever(index, FakeEmbedder(), mode="keyword").search("q", k=2)
    assert [h.chunk.id for h in result] == ["c", "a"]
    assert index.calls == [("keyword", 2)]


def test_hybrid_mode_fuses_both() -> None:
    result = Retriever(FakeIndex(hits(A, B), hits(C, A)), FakeEmbedder(), mode="hybrid").search(
        "q", 3
    )
    assert [h.chunk.id for h in result] == ["a", "c", "b"]


def test_rerank_reorders_candidates() -> None:
    retriever = Retriever(FakeIndex(hits(B, A), hits(B)), FakeEmbedder(), KeywordReranker())
    result = retriever.search("stomach bleeding", k=2)
    assert [h.chunk.id for h in result] == ["a", "b"]  # reranker prefers the relevant passage
    assert result[0].score == 2.0  # scores are now reranker scores


def test_rerank_mode_requires_a_reranker() -> None:
    with pytest.raises(ValueError, match="needs a reranker"):
        Retriever(FakeIndex([], []), FakeEmbedder(), mode="hybrid_rerank")
