"""Structured original reuse, full-corpus preservation and fail-closed reviews."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding
from qdrant_client import QdrantClient

from app.services.corpus import CorpusCatalog
from app.services.ingestion import (
    ExternalDataTransferRequired,
    IngestionConfig,
    IngestionService,
)
from tests.unit.test_corpus import write_pdf


def fixture_service(tmp_path: Path) -> tuple[IngestionService, dict]:
    data = tmp_path / "data"
    data.mkdir()
    write_pdf(data / "manual.pdf")
    (data / "other.html").write_text("<h1>Separate source</h1><p>Untouched original.</p>", encoding="utf-8")
    (data / "notes.txt").write_text("Independent notes retained.", encoding="utf-8")
    service = IngestionService(IngestionConfig(data_dir=data, docstore_path=tmp_path / "store.json",
                                              collection_name="review_test", embedding_dimensions=8,
                                              chunk_size=128, chunk_overlap=8),
                               client=QdrantClient(":memory:"), embed_model=MockEmbedding(embed_dim=8))
    service.ingest()
    # Reopen the persisted store: a live parser cache cannot satisfy this test.
    reopened = IngestionService(service.config, client=service._client, embed_model=MockEmbedding(embed_dim=8))
    original = next(d for d in reopened.docstore.docs.values() if d.metadata.get("page") == 3)
    image = next(b for b in original.metadata["evidence_blocks"] if b["kind"] == "visual_unextracted")
    record = {k: original.metadata[k] for k in ("source_hash", "source_id", "location_id", "page", "product", "version")}
    record.update(bbox=image["bbox"], image_sha256=image["attributes"]["image_sha256"],
                  region_id=image["block_id"], relative_path="manual.pdf",
                  kind="formula", text="E = 15 kPa", approval_status="approved", reviewer="independent-fixture",
                  reviewed_at="2026-10-16T00:00:00Z", review_id="literal-review-1")
    return reopened, record


def catalog_for(service: IngestionService) -> CorpusCatalog:
    return CorpusCatalog.load(service.config.docstore_path.with_suffix(".corpus.json"))


def points(service: IngestionService) -> dict:
    values, _ = service._client.scroll(service.config.collection_name, limit=100, with_payload=True)
    return {v.id: v.payload for v in values}


def test_review_reuses_persisted_originals_and_retains_every_other_page_source(tmp_path, monkeypatch):
    service, record = fixture_service(tmp_path)
    before_catalog = catalog_for(service)
    before_docs = {k: d.hash for k, d in service.docstore.docs.items()}
    before_points = points(service)
    monkeypatch.setattr("app.services.ingestion._load_one_file", lambda *a, **k: pytest.fail("No PDF/source reparse"))

    result = service.ingest_reviewed_transcriptions([record])

    assert result.source_files == 3
    assert result.changed_documents == result.loaded_documents == 1
    updated = next(d for d in service.docstore.docs.values() if d.metadata.get("page") == 3)
    assert isinstance(updated, Document) and updated.text.endswith("E = 15 kPa")
    assert set(service.docstore.docs) == set(before_docs)
    assert all(d.hash == before_docs[k] for k, d in service.docstore.docs.items() if k != updated.id_)
    assert all(points(service)[k] == payload for k, payload in before_points.items()
               if payload.get("doc_id") != updated.id_)
    after = catalog_for(service)
    assert after.sources == before_catalog.sources
    assert after.locations == before_catalog.locations  # Includes the blank physical page.
    assert len(after.blocks) == len(before_catalog.blocks) + 1
    assert sum(b["attributes"].get("method") == "reviewed_visual_transcription" for b in after.blocks) == 1
    assert any(r["kind"] == "transcription_of" for r in after.relations)
    assert all(b in after.blocks for b in before_catalog.blocks if b["location_id"] != record["location_id"])

    # Idempotency persists across processes; no collection or pipeline call.
    reopened = IngestionService(service.config, client=service._client, embed_model=MockEmbedding(embed_dim=8))
    monkeypatch.setattr(reopened, "_ensure_collection", lambda: pytest.fail("No-op must not touch Qdrant"))
    repeated = reopened.ingest_reviewed_transcriptions([record, deepcopy(record)])
    assert repeated.changed_documents == repeated.indexed_nodes == 0
    assert repeated.unchanged_documents == 1
    assert catalog_for(reopened).as_dict() == after.as_dict()


@pytest.mark.parametrize("change", ["unrelated_source", "removed_source", "new_source", "catalog_hash", "lost_original",
                                    "lost_blank_page", "corrupt_original_text"])
def test_review_rejects_stale_or_incomplete_corpus_before_vector_changes(tmp_path, monkeypatch, change):
    service, record = fixture_service(tmp_path)
    if change == "unrelated_source":
        (service.config.data_dir / "notes.txt").write_text("Changed source bytes", encoding="utf-8")
    elif change == "removed_source":
        (service.config.data_dir / "notes.txt").unlink()
    elif change == "new_source":
        (service.config.data_dir / "new.html").write_text("New source", encoding="utf-8")
    elif change == "catalog_hash":
        catalog = catalog_for(service)
        catalog.sources[0]["source_hash"] = "0" * 64
        catalog.persist(service.config.docstore_path.with_suffix(".corpus.json"))
    elif change == "lost_blank_page":
        catalog = catalog_for(service)
        catalog.locations = [loc for loc in catalog.locations if loc.get("page") != 2]
        catalog.persist(service.config.docstore_path.with_suffix(".corpus.json"))
    elif change == "corrupt_original_text":
        original = next(d for d in service.docstore.docs.values() if d.metadata.get("page") == 1)
        original.set_content("Corrupted cache content")
        service.docstore.add_documents([original], allow_update=True)
    else:
        original = next(d for d in service.docstore.docs.values() if d.metadata.get("page") == 1)
        service.docstore.delete_document(original.id_)
    monkeypatch.setattr(service, "_ensure_collection", lambda: pytest.fail("Validation must precede Qdrant"))
    with pytest.raises(ValueError):
        service.ingest_reviewed_transcriptions([record])


@pytest.mark.parametrize("field,value", [("bbox", [1, 2, 3, 4]), ("image_sha256", "0" * 64),
                                         ("version", "invented"), ("source_id", "wrong"),
                                         ("location_id", "wrong"), ("reviewer", ""),
                                         ("region_id", "wrong"), ("relative_path", "other.pdf")])
def test_review_rejects_wrong_exact_binding_before_mutation(tmp_path, monkeypatch, field, value):
    service, record = fixture_service(tmp_path)
    snapshot = deepcopy(service.docstore.to_dict())
    record[field] = value
    monkeypatch.setattr(service, "_ensure_collection", lambda: pytest.fail("Invalid review cannot reach Qdrant"))
    with pytest.raises(ValueError):
        service.ingest_reviewed_transcriptions([record])
    assert service.docstore.to_dict() == snapshot


def test_pending_records_are_ignored_and_conflicting_approved_revision_is_rejected(tmp_path, monkeypatch):
    service, record = fixture_service(tmp_path)
    pending = dict(record, approval_status="pending_review", text="UNAPPROVED")
    assert service.ingest_reviewed_transcriptions([pending]).changed_documents == 0
    service.ingest_reviewed_transcriptions([record])
    snapshot = deepcopy(service.docstore.to_dict())
    monkeypatch.setattr(service, "_ensure_collection", lambda: pytest.fail("Conflicting review cannot reach Qdrant"))
    with pytest.raises(ValueError, match="Conflicting approved"):
        service.ingest_reviewed_transcriptions([dict(record, text="Conflicting text")])
    assert service.docstore.to_dict() == snapshot


def test_review_failed_embedding_restores_original_store_catalog_and_retry(tmp_path, monkeypatch):
    service, record = fixture_service(tmp_path)
    snapshot = deepcopy(service.docstore.to_dict())
    catalog_before = catalog_for(service).as_dict()
    splitter_pipeline = service._pipeline_or_create()
    def fail_after_upsert_mutation(pipeline, **kwargs):
        pipeline.docstore.delete_document(kwargs["documents"][0].id_, raise_error=False)
        raise RuntimeError("fixture provider failure")
    with monkeypatch.context() as scoped:
        scoped.setattr(type(splitter_pipeline), "run", fail_after_upsert_mutation)
        with pytest.raises(RuntimeError, match="fixture provider failure"):
            service.ingest_reviewed_transcriptions([record])
    assert service.docstore.to_dict() == snapshot
    assert catalog_for(service).as_dict() == catalog_before
    reopened = IngestionService(service.config, client=service._client, embed_model=MockEmbedding(embed_dim=8))
    assert reopened.docstore.to_dict() == snapshot
    assert reopened.ingest_reviewed_transcriptions([record]).changed_documents == 1


def test_review_respects_external_transfer_permission(tmp_path):
    service, record = fixture_service(tmp_path)
    service._requires_external_data_transfer = True
    with pytest.raises(ExternalDataTransferRequired):
        service.ingest_reviewed_transcriptions([record])
