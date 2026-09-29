"""Settings from environment variables, so the same image runs locally and on AWS.

The answering model uses judgekit's ModelConfig, so any provider works
(Bedrock, Anthropic API, or OpenAI-compatible). Keys are read from the env var
named in RXCITE_API_KEY_ENV, never from files.
"""

import os
from dataclasses import dataclass

from judgekit.providers import DEFAULT_JUDGE_MODEL_ID, ModelConfig

from rxcite.embeddings import RERANK_MODEL
from rxcite.retrieval import MODES, Mode


@dataclass(frozen=True)
class Settings:
    database_url: str
    mode: Mode
    top_k: int
    refuse_below: float | None
    rerank_model: str
    llm: ModelConfig


def _optional_float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def load_settings() -> Settings:
    env = os.environ
    mode = env.get("RXCITE_MODE", "vector")  # best measured mode (see README, Stage 3)
    if mode not in MODES:
        raise ValueError(f"RXCITE_MODE must be one of {MODES}, got {mode!r}")
    return Settings(
        database_url=env.get(
            "RXCITE_DATABASE_URL", "postgresql://rxcite:rxcite@localhost:5432/rxcite"
        ),
        mode=mode,
        top_k=int(env.get("RXCITE_TOP_K", "5")),
        # Chosen in Stage 4 on a calibration split; see results/stage4_report.txt.
        refuse_below=_optional_float(env.get("RXCITE_REFUSE_BELOW", "0.74")),
        rerank_model=env.get("RXCITE_RERANK_MODEL", RERANK_MODEL),
        llm=ModelConfig(
            provider=env.get("RXCITE_PROVIDER", "bedrock"),
            model=env.get("RXCITE_MODEL", DEFAULT_JUDGE_MODEL_ID),
            base_url=env.get("RXCITE_BASE_URL") or None,
            api_key_env=env.get("RXCITE_API_KEY_ENV") or None,
            region=env.get("RXCITE_REGION") or None,
            max_tokens=int(env.get("RXCITE_MAX_TOKENS", "400")),
        ),
    )
