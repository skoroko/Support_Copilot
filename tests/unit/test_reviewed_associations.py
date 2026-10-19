"""Reviewed definition links are explicit source evidence, never adjacency guesses."""
from __future__ import annotations

from copy import deepcopy

import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import MetadataMode

from app.services.ingestion import IngestionService
from app.services.related_evidence import RelatedEvidenceIndex
from app.services.structured_readers import EvidenceSplitter
from tests.unit.test_reviewed_ingestion import catalog_for, fixture_service


def associated_record(service, record):
    original = next(d for d in service.docstore.docs.values() if d.metadata.get("page") == record["page"])
    target = next(b for b in original.metadata["evidence_blocks"]
                  if b["kind"] in {"paragraph", "definition", "formula_candidate"} and b["text"])
    relation = {key: original.metadata[key] for key in
                ("source_id", "source_hash", "location_id", "page", "product", "version")}
    relation.update(kind="defines", to_id=target["block_id"], approval_status="approved",
                    review_scope="explicit_source_definition_link", review_id="separate-definition-review",
                    reviewer="source-association-fixture-reviewer", reviewed_at="2026-10-16T01:00:00Z",
                    proof={"basis": "explicit_definition", "formula_quote": record["text"], "target_quote": target["text"]})
    return dict(record, reviewed_relations=[relation]), target


class CachedOnlyEmbedding(MockEmbedding):
    allowed: set[str]

    def _get_text_embedding(self, text):
        assert text in self.allowed, "Association-only updates must reuse identical embedding text"
        return super()._get_text_embedding(text)


def test_existing_literal_adds_reviewed_edges_without_text_change_and_resolves_inverse(tmp_path, monkeypatch):
    service, record = fixture_service(tmp_path)
    service.ingest_reviewed_transcriptions([record])
    before_catalog = catalog_for(service)
    before_docs = {key: document.hash for key, document in service.docstore.docs.items()}
    before_nodes = EvidenceSplitter(chunk_size=128, chunk_overlap=8)(list(service.docstore.docs.values()))
    allowed = {node.get_content(metadata_mode=MetadataMode.EMBED) for node in before_nodes}
    service._embed_model = CachedOnlyEmbedding(embed_dim=8, allowed=allowed)
    service._pipeline = None
    reviewed, native = associated_record(service, record)
    original = next(d for d in service.docstore.docs.values() if d.metadata.get("page") == record["page"])
    original_text = original.text
    result = service.ingest_reviewed_transcriptions([reviewed])
    assert result.changed_documents == 1
    updated = service.docstore.get_document(original.id_)
    assert updated.text == original_text
    assert len(updated.metadata["evidence_blocks"]) == len(original.metadata["evidence_blocks"])
    after_nodes = EvidenceSplitter(chunk_size=128, chunk_overlap=8)(list(service.docstore.docs.values()))
    assert {n.id_: n.get_content(metadata_mode=MetadataMode.EMBED) for n in before_nodes} == {
        n.id_: n.get_content(metadata_mode=MetadataMode.EMBED) for n in after_nodes}
    assert all(document.hash == before_docs[key] for key, document in service.docstore.docs.items() if key != original.id_)
    after_catalog = catalog_for(service)
    assert after_catalog.sources == before_catalog.sources
    assert after_catalog.locations == before_catalog.locations
    assert len(after_catalog.blocks) == len(before_catalog.blocks)
    edges = [r for r in after_catalog.relations if r["kind"] == "defines"]
    assert len(edges) == 1 and edges[0]["attributes"]["semantic_verified"] is False
    assert edges[0]["attributes"]["review_id"] != record["review_id"]
    index = RelatedEvidenceIndex(service.config.data_dir, service.config.docstore_path.with_suffix(".corpus.json"))
    metadata = {**updated.metadata, "block_id": native["block_id"]}
    related = index.resolve(metadata, native["text"], question="Which literal expression defines this parameter?")
    definitions = [r for r in related if r["metadata"]["relation"] == "inverse_defines"]
    assert len(definitions) == 1 and definitions[0]["text"] == record["text"]
    edge = next(r for r in definitions[0]["metadata"]["block_relations"] if r["kind"] == "defines")
    assert edge["attributes"]["proof"] == reviewed["reviewed_relations"][0]["proof"]
    reopened = IngestionService(service.config, client=service._client, embed_model=CachedOnlyEmbedding(embed_dim=8, allowed=allowed))
    monkeypatch.setattr(reopened, "_ensure_collection", lambda: pytest.fail("Identical relation is a no-op"))
    assert reopened.ingest_reviewed_transcriptions([reviewed]).changed_documents == 0


def test_new_literal_can_receive_separately_approved_edges_in_the_same_transaction(tmp_path):
    service, record = fixture_service(tmp_path)
    reviewed, _ = associated_record(service, record)
    service.ingest_reviewed_transcriptions([reviewed])
    assert sum(b["kind"] == "formula" for b in catalog_for(service).blocks) == 1
    assert sum(r["kind"] == "defines" for r in catalog_for(service).relations) == 1


@pytest.mark.parametrize("field,value", [("source_hash", "wrong"), ("source_id", "wrong"),
                                         ("location_id", "wrong"), ("page", 1), ("product", "other"),
                                         ("version", "other"), ("reviewer", ""),
                                         ("review_id", "literal-review-1"),
                                         ("review_scope", "literal_image_transcription"),
                                         ("kind", "next_page")])
def test_relation_approval_requires_separate_scope_and_exact_source_page(tmp_path, monkeypatch, field, value):
    service, record = fixture_service(tmp_path)
    service.ingest_reviewed_transcriptions([record])
    reviewed, _ = associated_record(service, record)
    reviewed["reviewed_relations"][0][field] = value
    snapshot = deepcopy(service.docstore.to_dict())
    monkeypatch.setattr(service, "_ensure_collection", lambda: pytest.fail("Invalid association cannot reach Qdrant"))
    with pytest.raises(ValueError, match="same-page native"):
        service.ingest_reviewed_transcriptions([reviewed])
    assert service.docstore.to_dict() == snapshot


def test_literal_approval_never_infers_edges_and_pending_relation_is_ignored(tmp_path):
    service, record = fixture_service(tmp_path)
    service.ingest_reviewed_transcriptions([record])
    assert not any(r["kind"] == "defines" for r in catalog_for(service).relations)
    reviewed, _ = associated_record(service, record)
    reviewed["reviewed_relations"][0]["approval_status"] = "pending"
    assert service.ingest_reviewed_transcriptions([reviewed]).changed_documents == 0
    assert not any(r["kind"] == "defines" for r in catalog_for(service).relations)


@pytest.mark.parametrize("change", ["cross_page_target", "image_target", "missing_quote", "conflict"])
def test_cross_page_non_native_unproven_or_revised_targets_are_rejected(tmp_path, monkeypatch, change):
    service, record = fixture_service(tmp_path)
    service.ingest_reviewed_transcriptions([record])
    reviewed, _ = associated_record(service, record)
    row = reviewed["reviewed_relations"][0]
    if change == "cross_page_target":
        other = next(d for d in service.docstore.docs.values() if d.metadata.get("page") == 1)
        row["to_id"] = next(b["block_id"] for b in other.metadata["evidence_blocks"] if b["text"])
    elif change == "image_target":
        row["to_id"] = record["region_id"]
    elif change == "missing_quote":
        row["proof"]["target_quote"] = "not present in the source"
    else:
        service.ingest_reviewed_transcriptions([reviewed])
        row["review_id"] = "unreviewed-replacement-id"
    snapshot = deepcopy(service.docstore.to_dict())
    monkeypatch.setattr(service, "_ensure_collection", lambda: pytest.fail("Invalid relation cannot reach Qdrant"))
    with pytest.raises(ValueError):
        service.ingest_reviewed_transcriptions([reviewed])
    assert service.docstore.to_dict() == snapshot


def test_resolver_rejects_reviewed_edge_tampered_to_a_different_page(tmp_path):
    service, record = fixture_service(tmp_path)
    reviewed, target = associated_record(service, record)
    service.ingest_reviewed_transcriptions([reviewed])
    catalog = catalog_for(service)
    transcript = next(b for b in catalog.blocks if b["kind"] == "formula")
    edge = next(r for r in transcript["relations"] if r["kind"] == "defines")
    edge["to_id"] = next(b["block_id"] for b in catalog.blocks
                          if b["location_id"] != transcript["location_id"] and b["text"])
    path = service.config.docstore_path.with_suffix(".corpus.json")
    catalog.persist(path)
    original = next(d for d in service.docstore.docs.values() if d.metadata.get("page") == record["page"])
    assert RelatedEvidenceIndex(service.config.data_dir, path).resolve(
        {**original.metadata, "block_id": target["block_id"]}, target["text"], question="Which definition?") == []
