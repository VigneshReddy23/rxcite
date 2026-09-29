import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from rxcite import ingest

LABEL: dict[str, Any] = {
    "set_id": "abc-123",
    "openfda": {"generic_name": ["IBUPROFEN"], "brand_name": ["Advil"]},
    "warnings": ["WARNINGS Allergy alert: Ibuprofen may cause a severe allergic reaction."],
    "storage_and_handling": ["Store at 20-25°C."],
    "package_label_principal_display_panel": ["PRINCIPAL DISPLAY PANEL artwork text"],
}


def test_drug_name_combines_generic_and_brand() -> None:
    assert ingest.drug_name(LABEL) == "Ibuprofen (Advil)"
    assert (
        ingest.drug_name({"openfda": {"generic_name": ["ASPIRIN"], "brand_name": ["aspirin"]}})
        == "Aspirin"
    )
    assert ingest.drug_name({}) == "Unknown Drug"


def test_clean_collapses_whitespace_and_drops_heading() -> None:
    assert ingest.clean("WARNINGS  Allergy\n alert: text") == "Allergy alert: text"
    assert ingest.clean("5 WARNINGS AND PRECAUTIONS Serious risk") == "Serious risk"
    assert ingest.clean("NSAIDs may cause bleeding") == "NSAIDs may cause bleeding"


def test_split_words_overlaps() -> None:
    words = " ".join(str(i) for i in range(400))
    parts = ingest.split_words(words, size=180, overlap=30)
    assert [len(p.split()) for p in parts] == [180, 180, 100]
    assert parts[1].split()[0] == "150"  # starts 30 words before the previous end
    assert ingest.split_words("short text") == ["short text"]
    assert ingest.split_words("   ") == []


def test_chunk_label_keeps_useful_sections_only() -> None:
    chunks = ingest.chunk_label(LABEL)
    assert [c.id for c in chunks] == ["abc-123:warnings:0", "abc-123:storage_and_handling:0"]
    assert chunks[0].section == "Warnings"
    assert chunks[0].text.startswith("Allergy alert")
    assert chunks[0].drug == "Ibuprofen (Advil)"


def test_get_json_retries_server_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def fake_urlopen(request: Any, timeout: int) -> Any:
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.HTTPError("u", 503, "busy", {}, None)  # type: ignore[arg-type]
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert ingest._get_json("http://x", backoff_seconds=0) == {"ok": True}
    assert len(attempts) == 3


def test_get_json_does_not_retry_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def fake_urlopen(request: Any, timeout: int) -> Any:
        attempts.append(1)
        raise urllib.error.HTTPError("u", 404, "no match", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert ingest.latest_label("NOT A DRUG") is None
    assert len(attempts) == 1


def test_download_labels_skips_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "top_generic_names", lambda n: ["A", "B", "C"])

    def fake_latest(name: str) -> dict[str, Any] | None:
        if name == "B":
            raise urllib.error.URLError("down")
        return None if name == "C" else LABEL

    monkeypatch.setattr(ingest, "latest_label", fake_latest)
    out = tmp_path / "raw" / "labels.jsonl"
    assert ingest.download_labels(3, out, pause_seconds=0) == 1
    assert [json.loads(line)["set_id"] for line in out.read_text().splitlines()] == ["abc-123"]
    assert list(ingest.load_labels(out))[0]["set_id"] == "abc-123"


def test_dailymed_url() -> None:
    assert ingest.dailymed_url("abc").endswith("setid=abc")
