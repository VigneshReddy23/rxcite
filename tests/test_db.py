"""Integration tests against a real Postgres + pgvector.

Run with a database available, e.g.:
  docker compose up -d db
  RXCITE_TEST_DATABASE_URL=postgresql://rxcite:rxcite@localhost:5432/rxcite pytest -m db
CI provides one as a service container. Skipped otherwise.
"""

import os
from collections.abc import Iterator

import psycopg
import pytest

from rxcite import db
from rxcite.embeddings import EMBED_DIM
from tests.fakes import chunk

URL = os.environ.get("RXCITE_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not URL, reason="RXCITE_TEST_DATABASE_URL not set"),
]


def one_hot(i: int) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[i] = 1.0
    return v


@pytest.fixture
def conn() -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    """A connection whose tables live in a throwaway `rxcite_test` schema.

    search_path makes unqualified names ("chunks") resolve there first, so these
    tests can never drop or modify real data in the default schema.
    """
    assert URL
    c = db.connect(URL)
    c.execute("DROP SCHEMA IF EXISTS rxcite_test CASCADE")
    c.execute("CREATE SCHEMA rxcite_test")
    c.execute("SET search_path TO rxcite_test, public")
    db.init_schema(c)
    db.upsert_chunks(
        c,
        [
            chunk("a", "Ibuprofen may cause severe stomach bleeding."),
            chunk("b", "Store at room temperature away from moisture.", drug="Loratadine"),
            chunk("c", "Acetaminophen overdose can cause liver damage.", drug="Acetaminophen"),
        ],
        [one_hot(0), one_hot(1), one_hot(2)],
    )
    yield c
    c.execute("DROP SCHEMA IF EXISTS rxcite_test CASCADE")
    c.close()


def test_vector_search_orders_by_cosine(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    near_b = one_hot(1)
    near_b[0] = 0.3
    hits = db.vector_search(conn, near_b, k=2)
    assert [h.chunk.id for h in hits] == ["b", "a"]
    assert hits[0].score > hits[1].score


def test_keyword_search_matches_any_word(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    hits = db.keyword_search(conn, "can ibuprofen cause liver problems?", k=5)
    assert {h.chunk.id for h in hits} == {"a", "c"}  # OR semantics: either word matches


def test_keyword_search_stopwords_only(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    assert db.keyword_search(conn, "the and of", k=5) == []


def test_tests_do_not_touch_the_default_schema(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    def public_table_oid() -> object:
        row = conn.execute("SELECT to_regclass('public.chunks')::oid").fetchone()
        return row[0] if row else None

    before = public_table_oid()
    conn.execute("DROP TABLE IF EXISTS chunks")  # resolves to rxcite_test.chunks
    assert public_table_oid() == before


def test_pooled_index_matches_direct_queries(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    from rxcite.retrieval import PooledIndex

    assert URL
    pool = db.make_pool(URL + "?options=-csearch_path%3Drxcite_test,public", max_size=2)
    try:
        index = PooledIndex(pool)
        assert [h.chunk.id for h in index.vector(one_hot(2), 1)] == ["c"]
        assert {h.chunk.id for h in index.keyword("liver", 5)} == {"c"}
    finally:
        pool.close()


def test_upsert_is_idempotent(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    db.upsert_chunks(conn, [chunk("a", "updated text")], [one_hot(0)])
    assert db.count_chunks(conn) == 3
