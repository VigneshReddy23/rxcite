"""Data shapes shared across rxcite."""

from pydantic import BaseModel


class Chunk(BaseModel):
    """One searchable passage: a piece of one section of one drug label."""

    id: str  # "<set_id>:<section>:<n>", stable across re-ingests
    set_id: str  # FDA label identifier; links back to DailyMed
    drug: str  # display name, e.g. "Ibuprofen (Advil)"
    section: str  # human-readable section, e.g. "Warnings"
    text: str


class Hit(BaseModel):
    """A chunk returned by search, with the score that ranked it."""

    chunk: Chunk
    score: float


class Citation(BaseModel):
    number: int  # the [n] used in the answer text
    chunk_id: str
    drug: str
    section: str
    url: str  # DailyMed page for the label


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    refused: bool
    retrieval_ms: float
    generation_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
