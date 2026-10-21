import json
from types import SimpleNamespace

import pytest

from scripts.export_index_fragments import adapt_records, export
from scripts.load_to_qdrant import load_documents, point_id


def records(count=100):
    return [SimpleNamespace(
        id=str(index), vector=[0.1, 0.2],
        payload={"_node_content": json.dumps({
            "text": f"Real fragment {index}",
            "metadata": {"source": "manual.pdf", "category": "manual", "created_at": "2026-09-01T00:00:00Z"},
        })},
    ) for index in range(count)]


def test_adapter_preserves_original_text_identity_vectors_and_loader_contract(tmp_path):
    source = records()
    documents, vectors = adapt_records(source)
    output = tmp_path / "fragments.json"
    output.write_text(json.dumps(documents), encoding="utf-8")
    assert load_documents(output) == documents
    assert documents[0]["text"] == "Real fragment 0"
    assert documents[0]["chunk_index"] == "0"
    assert vectors[0]["vector"] == source[0].vector
    assert len({point_id(document) for document in documents}) == 100


@pytest.mark.parametrize("damage", ["missing_date", "named_vector", "few_points"])
def test_adapter_rejects_missing_provenance_or_invalid_vectors(damage):
    source = records()
    if damage == "missing_date":
        source[0].payload = {"text": "real", "source": "manual.pdf", "category": "manual"}
    elif damage == "named_vector":
        source[0].vector = {"dense": [0.1]}
    else:
        source.pop()
    with pytest.raises(ValueError):
        adapt_records(source)


def test_export_uses_only_scroll_and_preserves_pagination():
    class Client:
        def __init__(self):
            self.calls = []

        def scroll(self, **kwargs):
            self.calls.append(kwargs)
            return (records()[:50], "next") if len(self.calls) == 1 else (records()[50:], None)

    client = Client()
    documents, _ = export(client, collection="existing", limit=100)
    assert len(documents) == 100
    assert client.calls[1]["offset"] == "next"
    assert all(call["with_vectors"] and call["with_payload"] for call in client.calls)
