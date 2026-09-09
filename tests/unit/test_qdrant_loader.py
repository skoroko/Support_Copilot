from __future__ import annotations

import json
from pathlib import Path

from scripts.load_to_qdrant import load_documents, point_id


ROOT = Path(__file__).resolve().parents[2]


def test_subject_corpus_has_120_meaningful_documents() -> None:
    path = ROOT / "data" / "support_documents.json"
    documents = load_documents(path)

    assert len(documents) == 120
    assert len({item["text"] for item in documents}) == 120
    assert {item["tenant_id"] for item in documents} == {"tenant-a", "tenant-b", "tenant-x"}
    assert any(item["archived"] for item in documents)
    assert any(not item["archived"] for item in documents)


def test_ids_are_deterministic_and_source_chunk_specific() -> None:
    documents = json.loads((ROOT / "data" / "support_documents.json").read_text(encoding="utf-8"))

    assert point_id(documents[0]) == point_id(dict(documents[0]))
    assert point_id(documents[0]) != point_id(documents[1])
