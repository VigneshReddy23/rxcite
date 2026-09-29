"""Offline stand-ins for the database, embedding model, reranker and LLM."""

from collections.abc import Sequence

from rxcite.models import Chunk, Hit


def chunk(cid: str, text: str, drug: str = "Ibuprofen", section: str = "Warnings") -> Chunk:
    return Chunk(id=cid, set_id=f"set-{cid}", drug=drug, section=section, text=text)


class FakeIndex:
    """Returns fixed rankings and records what it was asked."""

    def __init__(self, vector_hits: list[Hit], keyword_hits: list[Hit]) -> None:
        self.vector_hits = vector_hits
        self.keyword_hits = keyword_hits
        self.calls: list[tuple[str, int]] = []

    def vector(self, embedding: Sequence[float], k: int) -> list[Hit]:
        self.calls.append(("vector", k))
        return self.vector_hits[:k]

    def keyword(self, query: str, k: int) -> list[Hit]:
        self.calls.append(("keyword", k))
        return self.keyword_hits[:k]


class FakeEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0] for t in texts]


class KeywordReranker:
    """Scores a passage by how many query words it contains."""

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        words = set(query.lower().split())
        return [float(len(words & set(p.lower().split()))) for p in passages]


class ScriptedLLM:
    def __init__(self, reply: str, input_tokens: int = 100, output_tokens: int = 20) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self._in, self._out = input_tokens, output_tokens

    def complete(self, prompt: str, system: str = "") -> str:
        self.calls.append((prompt, system))
        self.input_tokens += self._in
        self.output_tokens += self._out
        return self.reply
