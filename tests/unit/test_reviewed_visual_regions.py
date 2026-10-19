from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest
from llama_index.core import Document

from app.services.corpus import (
    CorpusCatalog,
    apply_reviewed_transcriptions,
    file_hash,
    source_identity,
    stable_id,
)
from app.services.corpus_html import parse_html
from app.services.related_evidence import RelatedEvidenceIndex
from app.services.reviewed_visual_regions import (
    apply_reviewed_visual_region,
    canonical_sha256,
    native_snapshot,
    pdf_pixel_crop,
    validate_visual_locator,
)
from tests.unit.test_reviewed_ingestion import catalog_for, fixture_service


def vector_fixture(tmp_path, monkeypatch):
    service, _legacy = fixture_service(tmp_path)
    original = next(d for d in service.docstore.docs.values() if d.metadata.get("page") == 1)
    native = next(b for b in original.metadata["evidence_blocks"] if b["kind"] == "formula_candidate")
    bbox = native["bbox"]
    locator = {"kind": "pdf_render_region", "page": 1, "bbox": bbox, "render_sha256": "a" * 64,
        "recipe": {"renderer": "pdftoppm", "version": "fixture", "dpi": 300, "margin_points": 2,
            "pixel_crop": pdf_pixel_crop(bbox),
            "pixel_rounding": "floor min; ceil max; bbox expanded by margin_points before scaling",
            "recipe_version": "pdf_vector_pdftoppm300_margin2_v1"},
        "contributing_native_block_ids": [native["block_id"]],
        "contributing_native_blocks_sha256": canonical_sha256(native_snapshot([native]))}
    record = {key: original.metadata[key] for key in
              ("source_id", "source_hash", "location_id", "page", "product", "version")}
    record.update(relative_path="manual.pdf", bbox=bbox, visual_locator=locator, kind="formula", text="E=10 kPa",
        approval_status="approved", reviewer="literal-source-fixture", reviewed_at="2026-10-16T00:00:00Z",
        review_id="fixture-literal-review", verification_scope="literal_source_transcription_only", semantic_verified=False)
    monkeypatch.setattr("app.services.reviewed_visual_regions.render_pdf_region_sha256", lambda *a, **k: "a" * 64)
    return service, original, native, record


def test_vector_incremental_preserves_full_scope_and_originals_resolves_literal_derivation(tmp_path, monkeypatch):
    service, original, native, record = vector_fixture(tmp_path, monkeypatch)
    before = catalog_for(service)
    original_ids = set(service.docstore.docs)
    monkeypatch.setattr("app.services.ingestion._load_one_file", lambda *a, **k: pytest.fail("No full source reparse"))
    result = service.ingest_reviewed_transcriptions([record])
    assert result.changed_documents == 1 and result.source_files == len(before.sources)
    after = catalog_for(service)
    assert before.sources == after.sources and before.locations == after.locations
    assert set(service.docstore.docs) == original_ids and len(after.blocks) == len(before.blocks) + 1
    updated = service.docstore.get_document(original.id_)
    transcript = next(b for b in updated.metadata["evidence_blocks"] if b["kind"] == "formula")
    assert all(b in updated.metadata["evidence_blocks"] for b in original.metadata["evidence_blocks"])
    assert "image_sha256" not in transcript["attributes"]
    assert transcript["attributes"]["verification_scope"] == "literal_source_transcription_only"
    assert transcript["attributes"]["semantic_verified"] is False
    index = RelatedEvidenceIndex(service.config.data_dir, service.config.docstore_path.with_suffix(".corpus.json"))
    resolved = index.resolve({**updated.metadata, "block_id": native["block_id"]}, native["text"], question="What expression?")
    derived = [row for row in resolved if row["metadata"]["relation"] == "inverse_literal_render_of"]
    assert len(derived) == 1 and derived[0]["text"] == record["text"]
    monkeypatch.setattr(service, "_ensure_collection", lambda: pytest.fail("No-op does not touch vector store"))
    assert service.ingest_reviewed_transcriptions([record, deepcopy(record)]).changed_documents == 0
    assert catalog_for(service).as_dict() == after.as_dict()


def test_full_build_accepts_same_generic_sidecar_and_requires_original_data_root(tmp_path, monkeypatch):
    service, original, _native, record = vector_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="data root"):
        apply_reviewed_transcriptions([deepcopy(original)], [record])
    built = CorpusCatalog.build(service.config.data_dir, transcriptions=[record])
    assert any(b["kind"] == "formula" and b["text"] == record["text"] for b in built.blocks)
    pending = {**record, "approval_status": "pending_review"}
    assert not any(b["kind"] == "formula" for b in CorpusCatalog.build(service.config.data_dir, transcriptions=[pending]).blocks)


@pytest.mark.parametrize("change", ["source", "path", "bbox", "render_sha", "native_sha", "native_id", "page", "fake_image", "scope", "semantic"])
def test_vector_review_rejects_source_geometry_render_native_drift_before_mutation(tmp_path, monkeypatch, change):
    service, _original, _native, record = vector_fixture(tmp_path, monkeypatch)
    record = deepcopy(record)
    if change == "source":
        (service.config.data_dir / "manual.pdf").write_bytes(b"changed original")
    elif change == "path":
        record["relative_path"] = "../manual.pdf"
    elif change == "bbox":
        record["bbox"] = record["visual_locator"]["bbox"] = [1, 1, 2, 2]
    elif change == "render_sha":
        record["visual_locator"]["render_sha256"] = "b" * 64
    elif change == "native_sha":
        record["visual_locator"]["contributing_native_blocks_sha256"] = "b" * 64
    elif change == "native_id":
        record["visual_locator"]["contributing_native_block_ids"] = ["foreign-native-block"]
    elif change == "page":
        record["visual_locator"]["page"] = 3
    elif change == "fake_image":
        record["image_sha256"] = "a" * 64
    elif change == "scope":
        record["verification_scope"] = "engineering_verified"
    else:
        record["semantic_verified"] = True
    snapshot = deepcopy(service.docstore.to_dict())
    monkeypatch.setattr(service, "_ensure_collection", lambda: pytest.fail("Invalid review must not touch vectors"))
    with pytest.raises(ValueError):
        service.ingest_reviewed_transcriptions([record])
    assert service.docstore.to_dict() == snapshot


def test_literal_id_is_locator_bound_and_conflicting_revisions_rejected(tmp_path, monkeypatch):
    service, original, _native, record = vector_fixture(tmp_path, monkeypatch)
    a, b = deepcopy(original), deepcopy(original)
    b.metadata["evidence_blocks"].append({"block_id": "extra", "kind": "paragraph", "text": "", "ordinal": 500})
    apply_reviewed_visual_region(a, record, service.config.data_dir)
    apply_reviewed_visual_region(b, record, service.config.data_dir)
    assert a.metadata["evidence_blocks"][-1]["block_id"] == b.metadata["evidence_blocks"][-1]["block_id"]
    service.ingest_reviewed_transcriptions([record])
    with pytest.raises(ValueError, match="Conflicting approved visual"):
        service.ingest_reviewed_transcriptions([{**record, "text": "changed literal"}])


def test_runtime_derivation_rejects_catalog_native_drift_without_rendering(tmp_path, monkeypatch):
    service, original, native, record = vector_fixture(tmp_path, monkeypatch)
    service.ingest_reviewed_transcriptions([record])
    catalog = catalog_for(service)
    next(b for b in catalog.blocks if b["block_id"] == native["block_id"])["text"] = "changed native text"
    catalog.persist(service.config.docstore_path.with_suffix(".corpus.json"))
    monkeypatch.setattr("app.services.reviewed_visual_regions.render_pdf_region_sha256", lambda *a, **k: pytest.fail("No render per request"))
    index = RelatedEvidenceIndex(service.config.data_dir, service.config.docstore_path.with_suffix(".corpus.json"))
    assert index.resolve({**original.metadata, "block_id": native["block_id"]}, "changed native text", question="formula") == []


def chm_fixture(tmp_path, monkeypatch):
    source = tmp_path / "manual.chm"
    source.write_bytes(b"original CHM fixture bytes")
    member, asset = "/A/B/topic.htm", "/image.gif"
    sid, _ = source_identity(source, tmp_path)
    lid = stable_id("location", sid, member)
    raw = b'<p>Source topic</p><img src="../../image.gif">'
    image_bytes = b"GIF89a original asset bytes"

    class Archive:
        def __init__(self, path):
            self.path = path

        def files(self, **kwargs):
            return [{"name": member}, {"name": asset}]

        def read(self, name, **kwargs):
            return {member: raw, asset: image_bytes}[name]

    monkeypatch.setattr("pylibmspack.ChmArchive", Archive)
    parsed = parse_html(raw, lid)
    document = Document(text=parsed["text"], metadata={"source_id": sid, "source_hash": file_hash(source),
        "location_id": lid, "chm_member": member, "product": "gts_nx", "version": "unknown",
        "evidence_blocks": parsed["blocks"], "relations": parsed["relations"], "extraction_status": "text"})
    image = next(b for b in parsed["blocks"] if b["kind"] == "image")
    record = {key: document.metadata[key] for key in ("source_id", "source_hash", "location_id", "product", "version")}
    record.update(relative_path="manual.chm", kind="figure_transcription", text="Create Hinge | Apply",
        approval_status="approved", reviewer="literal-source-fixture", reviewed_at="2026-10-16T00:00:00Z",
        review_id="chm-source-literal-review", verification_scope="literal_source_transcription_only", semantic_verified=False,
        visual_locator={"kind": "chm_asset", "member": member, "asset_member": asset,
                        "asset_sha256": hashlib.sha256(image_bytes).hexdigest(), "source_image_block_id": image["block_id"]})
    return document, record


def test_chm_literal_uses_exact_original_archive_asset_without_pdf_geometry(tmp_path, monkeypatch):
    document, record = chm_fixture(tmp_path, monkeypatch)
    locator, original = validate_visual_locator(document, record, tmp_path)
    assert locator["kind"] == "chm_asset" and len(original) == 1
    assert apply_reviewed_visual_region(document, record, tmp_path)
    transcript = document.metadata["evidence_blocks"][-1]
    assert transcript["bbox"] is None and "page" not in transcript["attributes"]
    assert "image_sha256" not in transcript["attributes"]
    assert transcript["relations"] == [{"kind": "transcription_of", "to_id": original[0]["block_id"]}]
    assert not apply_reviewed_visual_region(document, record, tmp_path)


@pytest.mark.parametrize("change", ["member", "traversal", "asset_hash", "block", "href", "page", "fake_image", "source"])
def test_chm_review_rejects_crossmember_path_asset_or_source_conflicts(tmp_path, monkeypatch, change):
    document, record = chm_fixture(tmp_path, monkeypatch)
    if change == "member":
        record["visual_locator"]["member"] = "/other.htm"
    elif change == "traversal":
        record["visual_locator"]["asset_member"] = "/../image.gif"
    elif change == "asset_hash":
        record["visual_locator"]["asset_sha256"] = "0" * 64
    elif change == "block":
        record["visual_locator"]["source_image_block_id"] = "other"
    elif change == "href":
        document.metadata["evidence_blocks"][-1]["attributes"]["src"] = "../../../image.gif"
    elif change == "page":
        record["page"] = 1
    elif change == "fake_image":
        record["image_sha256"] = "0" * 64
    else:
        (tmp_path / "manual.chm").write_bytes(b"modified source")
    before = deepcopy(document.metadata)
    with pytest.raises(ValueError):
        apply_reviewed_visual_region(document, record, tmp_path)
    assert document.metadata == before
