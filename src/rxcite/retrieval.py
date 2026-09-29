"""Find the passages most likely to answer a question.

Four modes, so evaluation can compare them on the same questions:
  vector         semantic similarity only (embeddings + pgvector)
  keyword        exact-word matching only (Postgres full-text search)
  hybrid         both lists merged with Reciprocal Rank Fusion
  hybrid_rerank  hybrid, then a cross-encoder re-orders the top candidates
"""

from collections.abc import Sequence
from typing import Literal, Protocol

import psycopg

from rxcite import db
from rxcite.embeddings import Embedder, Reranker
from rxcite.models import Hit

Mode = Literal["vector", "keyword", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("vector", "keyword", "hybrid", "hybrid_rerank")

CANDIDATES = 30  # how many each search returns before fusion / reranking
RRF_K = 60  # standard constant from the RRF paper (Cormack et al., 2009)


class SearchIndex(Protocol):
    def vector(self, embedding: Sequence[float], k: int) -> list[Hit]: ...
    def keyword(self, query: str, k: int) -> list[Hit]: ...


class PostgresIndex:
    def __init__(self, conn: psycopg.Connection[tuple[object, ...]]) -> None:
        self.conn = conn

    def vector(self, embedding: Sequence[float], k: int) -> list[Hit]:
        return db.vector_search(self.conn, embedding, k)

    def keyword(self, query: str, k: int) -> list[Hit]:
        return db.keyword_search(self.conn, query, k)


def reciprocal_rank_fusion(*ranked_lists: list[Hit], k: int = RRF_K) -> list[Hit]:
    """Merge rankings: each list gives a chunk 1 / (k + rank). Scores add up.

    RRF uses only positions, not raw scores, so it can combine a cosine
    similarity (0-1) and a full-text rank (unbounded) without calibrating them.
    """
    fused: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.chunk.id, hit)
    ordered = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    return [Hit(chunk=first_seen[cid].chunk, score=fused[cid]) for cid in ordered]


class Retriever:
    def __init__(
        self,
        index: SearchIndex,
        embedder: Embedder,
        reranker: Reranker | None = None,
        mode: Mode = "hybrid_rerank",
    ) -> None:
        if mode == "hybrid_rerank" and reranker is None:
            raise ValueError("hybrid_rerank mode needs a reranker")
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.mode = mode

    def search(self, question: str, k: int = 5) -> list[Hit]:
        if self.mode == "keyword":
            return self.index.keyword(question, k)
        vector_hits = self.index.vector(self.embedder.embed([question])[0], CANDIDATES)
        if self.mode == "vector":
            return vector_hits[:k]
        fused = reciprocal_rank_fusion(vector_hits, self.index.keyword(question, CANDIDATES))
        if self.mode == "hybrid":
            return fused[:k]
        assert self.reranker is not None
        candidates = fused[:CANDIDATES]
        scores = self.reranker.scores(question, [h.chunk.text for h in candidates])
        reranked = sorted(
            (Hit(chunk=h.chunk, score=s) for h, s in zip(candidates, scores, strict=True)),
            key=lambda h: h.score,
            reverse=True,
        )
        return reranked[:k]
