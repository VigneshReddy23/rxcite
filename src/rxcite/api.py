"""HTTP API (FastAPI).

POST /ask     {"question": "..."}  ->  answer, citations, refused, timings
GET  /health  liveness check for Docker / load balancers
"""

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rxcite.models import Answer
from rxcite.service import RAGService


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def create_app(service: RAGService | None = None) -> FastAPI:
    """Build the app. Tests pass a service; production builds one from settings."""
    app = FastAPI(title="rxcite", description="Cited answers from FDA drug labels.")
    state: dict[str, Any] = {"service": service}

    def get_service() -> RAGService:
        if state["service"] is None:
            state["service"] = build_service()  # lazy: models load on first use
        svc: RAGService = state["service"]
        return svc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask")
    def ask(request: AskRequest) -> Answer:
        return get_service().ask(request.question)

    return app


def build_service() -> RAGService:  # pragma: no cover - wires real DB + models
    from judgekit.providers import make_provider

    from rxcite import db
    from rxcite.config import load_settings
    from rxcite.embeddings import CrossEncoderReranker, FastEmbedder
    from rxcite.retrieval import RERANK_MODES, PostgresIndex, Retriever

    settings = load_settings()
    reranker = (
        CrossEncoderReranker(settings.rerank_model) if settings.mode in RERANK_MODES else None
    )
    retriever = Retriever(
        PostgresIndex(db.connect(settings.database_url)), FastEmbedder(), reranker, settings.mode
    )
    return RAGService(retriever, make_provider(settings.llm), settings.top_k, settings.refuse_below)


app = create_app()
