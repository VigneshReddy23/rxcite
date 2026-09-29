"""Latency percentiles and cost per 1,000 queries from recorded answers.

Cost uses real token counts reported by the model API times the per-million
token prices passed in (Bedrock rate card for the model used). Embeddings and
search run locally, so they add no API cost. Questions refused by the
retrieval-confidence threshold cost nothing (no LLM call).
"""

from collections.abc import Sequence

import numpy as np

from rxcite.answer_eval import AnswerRecord


def percentiles(values: Sequence[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}


def latency_summary(records: Sequence[AnswerRecord]) -> dict[str, dict[str, float]]:
    answered = [r for r in records if r.answer.generation_ms > 0]
    return {
        "retrieval_ms": percentiles([r.answer.retrieval_ms for r in records]),
        "generation_ms": percentiles([r.answer.generation_ms for r in answered]),
        "total_ms": percentiles([r.answer.retrieval_ms + r.answer.generation_ms for r in answered]),
    }


def cost_per_1k(
    records: Sequence[AnswerRecord], input_per_million: float, output_per_million: float
) -> dict[str, float]:
    """Mean tokens per LLM call and USD per 1,000 answered questions."""
    answered = [r for r in records if r.answer.generation_ms > 0]
    tin = float(np.mean([r.answer.input_tokens for r in answered]))
    tout = float(np.mean([r.answer.output_tokens for r in answered]))
    per_query = (tin * input_per_million + tout * output_per_million) / 1_000_000
    return {"input_tokens": tin, "output_tokens": tout, "usd_per_1k": per_query * 1000}
