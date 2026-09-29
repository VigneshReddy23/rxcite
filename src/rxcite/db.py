"""Postgres storage: pgvector for semantic search, full-text search for keywords.

One table holds everything. Two indexes make both searches fast:
  - HNSW on the embedding column (approximate nearest-neighbour, cosine)
  - GIN on a generated tsvector column (Postgres full-text search)
"""

from collections.abc import Sequence
from typing import cast

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from rxcite.embeddings import EMBED_DIM
from rxcite.models import Chunk, Hit

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id        text PRIMARY KEY,
    set_id    text NOT NULL,
    drug      text NOT NULL,
    section   text NOT NULL,
    text      text NOT NULL,
    embedding vector({EMBED_DIM}) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (
                  to_tsvector('english', drug || ' ' || section || ' ' || text)
              ) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
"""

COLUMNS = "id, set_id, drug, section, text"


def connect(url: str) -> psycopg.Connection[tuple[object, ...]]:
    conn = psycopg.connect(url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def make_pool(url: str, max_size: int = 10) -> ConnectionPool:
    """A pool of ready connections, so concurrent API requests don't queue on one.

    Each new connection gets the pgvector type registered before use.
    """
    return ConnectionPool(
        url,
        min_size=1,
        max_size=max_size,
        kwargs={"autocommit": True},
        configure=register_vector,
        open=True,
    )


def init_schema(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    conn.execute(SCHEMA)


def upsert_chunks(
    conn: psycopg.Connection[tuple[object, ...]],
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO chunks ({COLUMNS}, embedding) VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding",
            [
                (c.id, c.set_id, c.drug, c.section, c.text, np.array(e, dtype=np.float32))
                for c, e in zip(chunks, embeddings, strict=True)
            ],
        )


def count_chunks(conn: psycopg.Connection[tuple[object, ...]]) -> int:
    row = conn.execute("SELECT count(*) FROM chunks").fetchone()
    return cast(int, row[0]) if row else 0


def all_chunks(
    conn: psycopg.Connection[tuple[object, ...]],
) -> list[tuple[Chunk, list[float]]]:
    """Every chunk with its stored embedding (for export to another database)."""
    rows = conn.execute(f"SELECT {COLUMNS}, embedding FROM chunks ORDER BY id").fetchall()
    return [
        (
            Chunk(
                id=str(r[0]), set_id=str(r[1]), drug=str(r[2]), section=str(r[3]), text=str(r[4])
            ),
            [float(x) for x in r[5].to_list()],  # type: ignore[attr-defined]
        )
        for r in rows
    ]


def _hits(rows: list[tuple[object, ...]]) -> list[Hit]:
    return [
        Hit(
            chunk=Chunk(
                id=str(r[0]), set_id=str(r[1]), drug=str(r[2]), section=str(r[3]), text=str(r[4])
            ),
            score=float(r[5]),  # type: ignore[arg-type]
        )
        for r in rows
    ]


def vector_search(
    conn: psycopg.Connection[tuple[object, ...]], embedding: Sequence[float], k: int
) -> list[Hit]:
    """Nearest chunks by cosine similarity (1 = identical meaning)."""
    vec = np.array(embedding, dtype=np.float32)
    rows = conn.execute(
        f"SELECT {COLUMNS}, 1 - (embedding <=> %s) AS score FROM chunks "
        "ORDER BY embedding <=> %s LIMIT %s",
        (vec, vec, k),
    ).fetchall()
    return _hits(rows)


def keyword_search(conn: psycopg.Connection[tuple[object, ...]], query: str, k: int) -> list[Hit]:
    """Full-text search, matching ANY query word (OR), ranked by ts_rank_cd.

    plainto_tsquery joins words with AND, which fails for natural questions
    ("can I take X with Y while pregnant"), so we swap & for | first.
    """
    rows = conn.execute(
        f"""
        WITH q AS (
            SELECT replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery AS query
        )
        SELECT {COLUMNS}, ts_rank_cd(tsv, q.query) AS score
        FROM chunks, q
        WHERE q.query::text <> '' AND tsv @@ q.query
        ORDER BY score DESC
        LIMIT %s
        """,
        (query, k),
    ).fetchall()
    return _hits(rows)
