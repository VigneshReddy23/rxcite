"""Download FDA drug labels from openFDA and split them into searchable chunks.

openFDA data is public domain (CC0). There are ~260k label versions, mostly
near-duplicates (one drug is sold by many manufacturers), so we keep one
current label per generic drug name.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rxcite.models import Chunk

API = "https://api.fda.gov/drug/label.json"

# Label sections worth answering questions from, with readable names.
# (Package artwork, chemistry "description" and similar are left out.)
SECTIONS: dict[str, str] = {
    "boxed_warning": "Boxed Warning",
    "indications_and_usage": "Indications and Usage",
    "purpose": "Purpose",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "do_not_use": "Do Not Use",
    "warnings": "Warnings",
    "warnings_and_cautions": "Warnings and Precautions",
    "ask_doctor": "Ask a Doctor Before Use",
    "ask_doctor_or_pharmacist": "Ask a Doctor or Pharmacist",
    "when_using": "When Using This Product",
    "stop_use": "Stop Use and Ask a Doctor",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "pregnancy": "Pregnancy",
    "pregnancy_or_breast_feeding": "Pregnancy or Breast-Feeding",
    "pediatric_use": "Pediatric Use",
    "geriatric_use": "Geriatric Use",
    "overdosage": "Overdosage",
    "storage_and_handling": "Storage and Handling",
}

CHUNK_WORDS = 180  # small enough to be specific, big enough to keep context
OVERLAP_WORDS = 30  # so a sentence cut at a boundary still appears whole once


def _get_json(url: str, retries: int = 5, backoff_seconds: float = 5.0) -> dict[str, Any]:
    """GET JSON, retrying rate limits (429) and server errors (5xx) with backoff.

    Public APIs have outages; waiting 5s, 10s, 20s, 40s rides out short ones.
    Client errors (like 404 "no match") are raised immediately.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "rxcite/0.1"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data: dict[str, Any] = json.loads(response.read())
            return data
        except urllib.error.HTTPError as exc:
            if (exc.code != 429 and exc.code < 500) or attempt == retries:
                raise
        except urllib.error.URLError:
            if attempt == retries:
                raise
        time.sleep(backoff_seconds * 2**attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def top_generic_names(n: int) -> list[str]:
    """The n generic drug names with the most labels (a proxy for common drugs)."""
    url = f"{API}?count=openfda.generic_name.exact&limit={min(n, 1000)}"
    return [row["term"] for row in _get_json(url)["results"]][:n]


def latest_label(generic_name: str) -> dict[str, Any] | None:
    query = urllib.parse.urlencode(
        {
            "search": f'openfda.generic_name.exact:"{generic_name}"',
            "sort": "effective_time:desc",
            "limit": 1,
        }
    )
    try:
        results: list[dict[str, Any]] = _get_json(f"{API}?{query}")["results"]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # openFDA returns 404 for "no match"
            return None
        raise
    return results[0] if results else None


def download_labels(n: int, out: Path, pause_seconds: float = 0.3) -> int:
    """Save one current label for each of the top-n generic names as JSONL.

    The pause keeps us well under openFDA's 240 requests/minute limit for
    clients without an API key.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    with out.open("w", encoding="utf-8") as f:
        for name in top_generic_names(n):
            try:
                label = latest_label(name)
            except urllib.error.URLError as exc:  # HTTPError is a subclass
                print(f"  skipped {name!r}: {exc}")  # one bad drug shouldn't stop the run
                label = None
            if label is not None:
                f.write(json.dumps(label) + "\n")
                saved += 1
            time.sleep(pause_seconds)
    return saved


def load_labels(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def drug_name(label: dict[str, Any]) -> str:
    """Readable name: 'Ibuprofen (Advil)', or just the generic name."""
    fda = label.get("openfda", {})
    generic = (fda.get("generic_name") or ["Unknown drug"])[0].title()
    brand = (fda.get("brand_name") or [""])[0].strip()
    if brand and brand.lower() != generic.lower():
        return f"{generic} ({brand})"
    return str(generic)


def clean(text: str) -> str:
    """Collapse whitespace and drop a leading repeat of the section heading."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^\d*(\.\d+)*\s*[A-Z][A-Z &/,-]{3,}\s+(?=[A-Z])", "", text)


def split_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text] if words else []
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, len(words) - overlap, step)]


def chunk_label(label: dict[str, Any]) -> list[Chunk]:
    set_id = str(label.get("set_id", label.get("id", "unknown")))
    drug = drug_name(label)
    chunks = []
    for field, section in SECTIONS.items():
        text = clean(" ".join(label.get(field, [])))
        for n, piece in enumerate(split_words(text)):
            chunks.append(
                Chunk(
                    id=f"{set_id}:{field}:{n}",
                    set_id=set_id,
                    drug=drug,
                    section=section,
                    text=piece,
                )
            )
    return chunks


def dailymed_url(set_id: str) -> str:
    return f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
