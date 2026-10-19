"""Synthetic-only tests. Never open project holdout cases or contexts."""
import copy
import json
from argparse import Namespace

import pytest

from scripts import audit_holdout_references as audit


def test_default_metadata_does_not_open_sealed_content(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"split_counts": {"holdout": 2}, "group_counts": {"holdout": {"pdf_only": 2}}, "files_sha256": {"holdout_cases.json": "sealed", "holdout_contexts.json": "sealed"}}))
    (tmp_path / "holdout_manifest.json").write_text('{"references_pending_review":2}')
    (tmp_path / "holdout_cases.json").write_text("INVALID JSON: must never read")
    result = audit.audit(Namespace(suite=tmp_path, unlock_final_holdout=False))
    assert result["case_count"] == 2
    assert result["holdout_content_opened"] is False


def test_unlock_requires_explicit_root_freeze_before_reading_content(tmp_path):
    args = Namespace(suite=tmp_path, unlock_final_holdout=True, root_freeze_id=None, sealed_manifest_sha256=None, catalog_sha256=None)
    with pytest.raises(ValueError, match="root freeze"):
        audit.audit(args)


def test_literal_binding_requires_complete_case_sensitive_unique_quote():
    blocks = [{"block_id": "a", "text": "Alpha beta"}, {"block_id": "b", "text": " gamma."}]
    assert audit.literal_spans(blocks, "Alpha\nbeta gamma.") == [(blocks[0], "Alpha beta"), (blocks[1], "gamma.")]
    assert not audit.literal_spans(blocks, "alpha beta gamma.")
    assert not audit.literal_spans(blocks, "Alpha beta gamma. Missing tail")
    assert not audit.literal_spans(blocks * 2, "Alpha beta gamma.")


def fixture_review():
    case = {"id": "synthetic-1", "question": "What is the permitted operation?", "required_evidence_slots": [{"slot_id": "required-1", "file_sha256": "source-hash", "expected_visual_regions": []}]}
    block = {"block_id": "b1", "source_id": "s1", "location_id": "l1", "kind": "paragraph", "text": "The permitted operation is COPY.", "attributes": {}}
    catalog = {"blocks": [block], "sources": [{"source_id": "s1", "source_hash": "source-hash", "relative_path": "manual.pdf"}], "locations": [{"location_id": "l1", "page": 2}]}
    e = {"block_id": "b1", "location_id": "l1", "source_hash": "source-hash", "relative_path": "manual.pdf", "page": 2, "chm_member": None, "quote": block["text"], "quote_sha256": audit.digest(block["text"])}
    review = {"case_id": case["id"], "method": "machine_source_review", "human_expert_approval": False, "reviewer": "independent-machine-reader", "review_id": "review-1", "reviewed_at": "2026-10-16", "case_sha256": audit.digest(case), "catalog_sha256": "catalog-hash", "sealed_cases_file_sha256": "cases-hash", "reference_answer": "COPY is permitted.", "query_relative_coverage_verified": True, "sufficiency_rationale": "The question asks permitted operation; the exact block names COPY.", "acceptance_status": "pending_independent_review", "applicability_status": "source_specific_supported", "required_visual_status": "not_required", "claim_slots": [{"slot_id": "operation", "claim": "COPY is permitted", "alternatives": [{"evidence": [e]}]}]}
    return case, review, catalog


def validate(case, review, catalog):
    return audit.validate_review(case, review, catalog, {"source-hash"}, "catalog-hash", "cases-hash")


def test_separate_machine_review_binds_candidate_without_accepting_it():
    case, review, catalog = fixture_review()
    assert validate(case, review, catalog) == [("l1", "b1")]
    assert review["acceptance_status"] == "pending_independent_review"


@pytest.mark.parametrize("field,value", [("source_hash", "other"), ("relative_path", "other.pdf"), ("page", 3), ("location_id", "elsewhere"), ("block_id", "missing"), ("quote", "COPY"), ("quote_sha256", "stale"), ("chm_member", "/wrong.htm")])
def test_review_rejects_identity_and_quote_drift(field, value):
    case, review, catalog = fixture_review()
    review["claim_slots"][0]["alternatives"][0]["evidence"][0][field] = value
    with pytest.raises(ValueError):
        validate(case, review, catalog)


@pytest.mark.parametrize("field,value", [("human_expert_approval", True), ("reviewer", audit.BUILDER_ID), ("case_sha256", "other-case"), ("catalog_sha256", "other-catalog"), ("acceptance_status", "approved"), ("applicability_status", "unknown"), ("query_relative_coverage_verified", False), ("required_visual_status", "pending")])
def test_machine_review_cannot_approve_or_bypass_provenance(field, value):
    case, review, catalog = fixture_review()
    review[field] = value
    with pytest.raises(ValueError):
        validate(case, review, catalog)


def test_required_visual_cannot_be_satisfied_by_plain_text_or_declaration():
    case, review, catalog = fixture_review()
    case["required_evidence_slots"][0]["expected_visual_regions"] = ["formula"]
    review["case_sha256"] = audit.digest(case)
    review["required_visual_status"] = "approved_literal_bound"
    review["required_visual_evidence"] = {"required-1:formula": ["b1"]}
    with pytest.raises(ValueError, match="visual"):
        validate(case, review, catalog)
    attrs = {"method": "reviewed_visual_transcription", "verified": True, "review_id": "literal-approval", "reviewer": "independent-reader", "visual_locator": {"kind": "pdf_render_region", "render_sha256": "render"}}
    catalog["blocks"][0].update(kind="formula", attributes=attrs)
    review["claim_slots"][0]["alternatives"][0]["evidence"][0]["source_review"] = copy.deepcopy(attrs)
    assert validate(case, review, catalog)
    catalog["blocks"][0]["attributes"]["visual_locator"]["render_sha256"] = "drift"
    with pytest.raises(ValueError, match="visual"):
        validate(case, review, catalog)


def test_multisource_case_requires_every_original_source():
    case, review, catalog = fixture_review()
    with pytest.raises(ValueError, match="every required"):
        audit.validate_review(case, review, catalog, {"source-hash", "missing-second-source"}, "catalog-hash", "cases-hash")


def test_unlocked_synthetic_audit_never_invents_answer_or_changes_question(tmp_path, monkeypatch):
    case, _, catalog = fixture_review()
    case.update(source_group="pdf_only", task_type="concept", reference=None)
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source = source_dir / "manual.pdf"
    source.write_bytes(b"synthetic original bytes")
    original_hash = audit.file_sha(source)
    catalog["sources"][0]["source_hash"] = original_hash
    catalog["locations"][0].update(source_hash=original_hash, source_id="s1", product="synthetic", version="unknown")
    case["required_evidence_slots"][0].update(relative_path="manual.pdf", file_sha256=original_hash, physical_page=2, exact_quote=catalog["blocks"][0]["text"], applicability_status="pending_version_and_semantic_review")
    suite = tmp_path / "suite"
    suite.mkdir()
    for name, value in {"manifest.json": {}, "holdout_cases.json": [case], "holdout_contexts.json": {}}.items():
        (suite/name).write_text(json.dumps(value))
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    monkeypatch.setattr(audit, "load_suite", lambda *a, **k: ({}, [case], {}))
    args = Namespace(suite=suite, unlock_final_holdout=True, root_freeze_id="synthetic-final-freeze", sealed_manifest_sha256=audit.file_sha(suite/"manifest.json"), catalog_sha256=audit.file_sha(catalog_path), catalog=catalog_path, source_root=source_dir, output=tmp_path/"output", machine_reviews=None)
    original_cases_sha = audit.file_sha(suite/"holdout_cases.json")
    result = audit.audit(args)
    report = json.loads((args.output/"holdout_binding_contract.json").read_text())
    assert result["summary"] == {"pending": 1}
    assert report["cases"][0]["anchor_checks"][0]["exact_catalog_binding"] is True
    assert report["cases"][0]["reference_answer"] is None
    assert report["questions_modified"] == 0
    assert audit.file_sha(suite/"holdout_cases.json") == original_cases_sha
    with pytest.raises(FileExistsError):
        audit.audit(args)
