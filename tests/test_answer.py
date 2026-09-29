from fastapi.testclient import TestClient

from rxcite.answer import REFUSAL, build_prompt, cited_numbers, generate_answer
from rxcite.api import create_app
from rxcite.models import Hit
from rxcite.retrieval import Retriever
from rxcite.service import RAGService
from tests.fakes import FakeEmbedder, FakeIndex, ScriptedLLM, chunk

HITS = [
    Hit(chunk=chunk("a", "May cause stomach bleeding.", section="Warnings"), score=4.0),
    Hit(chunk=chunk("b", "Store at room temperature.", section="Storage"), score=1.0),
]


def test_prompt_numbers_sources_and_wraps_data() -> None:
    prompt = build_prompt("Is it safe?", HITS)
    assert "[1] Ibuprofen, Warnings:\nMay cause stomach bleeding." in prompt
    assert "[2] Ibuprofen, Storage:" in prompt
    assert prompt.endswith("<question>\nIs it safe?\n</question>")


def test_cited_numbers_in_order_and_valid_only() -> None:
    assert cited_numbers("A [2]. B [1][2]. C [7].", n_sources=2) == [2, 1]


def test_answer_with_citations() -> None:
    llm = ScriptedLLM("It may cause stomach bleeding [1].")
    answer = generate_answer("q", HITS, llm, retrieval_ms=12.0)
    assert not answer.refused
    assert [(c.number, c.chunk_id, c.section) for c in answer.citations] == [(1, "a", "Warnings")]
    assert answer.citations[0].url.endswith("setid=set-a")
    assert (answer.input_tokens, answer.output_tokens) == (100, 20)
    assert answer.retrieval_ms == 12.0 and answer.generation_ms >= 0


def test_low_confidence_refuses_without_calling_llm() -> None:
    llm = ScriptedLLM("should not be called")
    answer = generate_answer("q", HITS, llm, retrieval_ms=1.0, refuse_below=5.0)
    assert answer.refused and answer.answer == REFUSAL
    assert llm.calls == []


def test_no_hits_refuses() -> None:
    assert generate_answer("q", [], ScriptedLLM("x"), 1.0).refused


def test_llm_refusal_is_detected() -> None:
    answer = generate_answer("q", HITS, ScriptedLLM(REFUSAL), 1.0)
    assert answer.refused
    assert answer.citations == []


def make_client(reply: str = "Yes [1].") -> TestClient:
    retriever = Retriever(FakeIndex(HITS, []), FakeEmbedder(), mode="vector")
    return TestClient(create_app(RAGService(retriever, ScriptedLLM(reply), top_k=2)))


def test_api_health_and_ask() -> None:
    client = make_client()
    assert client.get("/health").json() == {"status": "ok"}
    body = client.post("/ask", json={"question": "Can ibuprofen cause bleeding?"}).json()
    assert body["answer"] == "Yes [1]."
    assert body["citations"][0]["chunk_id"] == "a"
    assert body["refused"] is False


def test_api_validates_input() -> None:
    assert make_client().post("/ask", json={"question": ""}).status_code == 422
