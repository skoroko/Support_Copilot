from __future__ import annotations

import hashlib
import json

from app.services.related_evidence import RelatedEvidenceIndex


def test_windows_payload_uses_hash_bound_catalog_path_on_another_host(tmp_path):
    data = tmp_path / "data"
    source = data / "chapter" / "manual.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source bytes for locator fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    path = tmp_path / "catalog.json"
    catalog = {"schema_version": 1, "sources": [{"source_id": "source", "source_hash": digest,
                                                 "relative_path": "chapter/manual.pdf"}],
               "locations": [], "blocks": [], "relations": []}
    path.write_text(json.dumps(catalog), encoding="utf-8")
    index = RelatedEvidenceIndex(data, path)
    metadata = {"source_id": "source", "source_hash": digest,
                "file_path": "Z:\\foreign-windows-host\\data\\chapter\\manual.pdf"}
    assert index.source_relative_path(metadata) == "chapter/manual.pdf"
    assert index.source_relative_path(dict(metadata, source_hash="wrong")) is None
    source.write_bytes(b"changed bytes")
    assert index.source_relative_path(metadata) is None


def test_portable_locator_never_accepts_catalog_traversal(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = tmp_path / "outside.pdf"
    source.write_bytes(b"outside bytes")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"schema_version": 1, "sources": [{"source_id": "source", "source_hash": digest,
                                                                 "relative_path": "../outside.pdf"}],
                                "locations": [], "blocks": [], "relations": []}), encoding="utf-8")
    assert RelatedEvidenceIndex(data, path).source_relative_path({"source_id": "source", "source_hash": digest}) is None
