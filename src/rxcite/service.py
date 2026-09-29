"""The question-answering pipeline: retrieve, then generate (or refuse)."""

import time

from judgekit.providers import Provider

from rxcite.answer import generate_answer
from rxcite.models import Answer
from rxcite.retrieval import Retriever


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        provider: Provider,
        top_k: int = 5,
        refuse_below: float | None = None,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.top_k = top_k
        self.refuse_below = refuse_below

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        hits = self.retriever.search(question, self.top_k)
        retrieval_ms = (time.perf_counter() - start) * 1000
        return generate_answer(question, hits, self.provider, retrieval_ms, self.refuse_below)
