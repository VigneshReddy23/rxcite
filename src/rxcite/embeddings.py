"""Text embeddings and cross-encoder reranking, both run locally (no API cost).

- Embeddings turn text into 384 numbers so similar meanings end up close
  ("stomach bleeding" ~ "GI hemorrhage").
- The reranker reads (question, passage) pairs together and scores relevance
  more accurately than embeddings, but is slower, so it only re-orders the top
  candidates.
"""

from collections.abc import Sequence
from typing import Protocol

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def scores(self, query: str, passages: Sequence[str]) -> list[float]: ...


class FastEmbedder:
    def __init__(self, model: str = EMBED_MODEL) -> None:
        from fastembed import TextEmbedding  # imported lazily: slow to load

        self.model = TextEmbedding(model)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(list(texts))]


class CrossEncoderReranker:
    def __init__(self, model: str = RERANK_MODEL) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model)

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, list(passages))]
