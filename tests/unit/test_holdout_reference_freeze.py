"""Synthetic30-case freezes only; real holdout files are never opened."""
import copy
import json

import pytest

from app.eval.holdout_reference_freeze import (
    _digest,
    _file_sha,
    load_holdout_reference_freeze,
)

IDENTITY = "f" * 64
CATALOG = "c" * 64


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fixture(tmp_path):
    suite, directory = tmp_path / "suite", tmp_path / "references"
    suite.mkdir(); directory.mkdir()
    cases, contexts, rows = [], {}, []
    for i in range(30):
        cid = f"synthetic-{i:03}"
        case = {"id": cid, "split": "holdout", "question": f"Synthetic question{i}", "reference_status": "pending_review", "expected_behavior": "answer", "required_evidence_slots": [{"file_sha256": "a"*64}]}
        cases.append(case)
        records = [{"source": {"relative_path": "manual.pdf", "file_sha256": "a"*64, "location_id": "location-"+cid, "physical_page": 2, "chm_member": None}, "text": f"Allowed action{i}.", "block_ids": ["block-"+cid]}]
        contexts[cid] = {"records": records, "fingerprint": _digest(records)}
        rows.append({"case_id": cid, "case_sha256": _digest(case), "candidate_case_sha256": _digest(case), "method": "machine_source_review", "human_expert_approval": False, "acceptance_status": "pending_independent_review", "sufficiency": "pending", "reviewer": "offline_holdout_binding_auditor_v1", "reference_answer": None, "claim_slots": []})
    write(suite/"holdout_cases.json", cases)
    write(suite/"holdout_contexts.json", contexts)
    sealed = {"split_counts": {"holdout": 30}, "files_sha256": {name: _file_sha(suite/name) for name in ("holdout_cases.json", "holdout_contexts.json")}}
    write(suite/"manifest.json", sealed)
    row = rows[0]; record = contexts[cases[0]["id"]]["records"][0]
    e = {"relative_path": "manual.pdf", "source_hash": "a"*64, "location_id": record["source"]["location_id"], "block_id": record["block_ids"][0], "page": 2, "chm_member": None, "quote": record["text"], "quote_sha256": _digest(record["text"])}
    row.update(sufficiency="machine_reviewed_sufficient_candidate", reviewer="independent-machine-reader", reference_answer="Allowed action0.", query_relative_review_sha256="d"*64, claim_slots=[{"slot_id": "required-action", "claim": "Allowed action0", "alternatives": [{"evidence": [e]}]}])
    metadata = {"split": "holdout", "root_freeze_id": IDENTITY, "catalog_sha256": CATALOG, "method": "machine_source_review", "human_expert_approval": False, "questions_modified": 0, "cases_sha256": _digest(cases), "sealed_manifest_sha256": _file_sha(suite/"manifest.json"), "sealed_cases_file_sha256": _file_sha(suite/"holdout_cases.json"), "sealed_contexts_file_sha256": _file_sha(suite/"holdout_contexts.json"), "machine_review_file_sha256": "e"*64, "summary": {"machine_reviewed_sufficient_candidate": 1, "pending": 29}}
    contract = {**copy.deepcopy(metadata), "cases": rows}
    write(directory/"holdout_binding_contract.json", contract)
    write(directory/"holdout_candidate_contexts.json", contexts)
    audit = copy.deepcopy(metadata)
    audit["artifacts"] = {name: _file_sha(directory/name) for name in ("holdout_binding_contract.json", "holdout_candidate_contexts.json")}
    write(directory/"manifest.json", audit)
    return suite, directory, cases, contexts, contract, audit


def load(data):
    suite, directory, cases, *_ = data
    return load_holdout_reference_freeze(directory, suite, cases, candidate_identity_sha256=IDENTITY, catalog_sha256=CATALOG)


def reseal(data):
    _, directory, _, contexts, contract, audit = data
    write(directory/"holdout_binding_contract.json", contract)
    write(directory/"holdout_candidate_contexts.json", contexts)
    audit["artifacts"] = {name: _file_sha(directory/name) for name in ("holdout_binding_contract.json", "holdout_candidate_contexts.json")}
    write(directory/"manifest.json", audit)


def test_valid_freeze_returns_two_values_without_promoting_pending(tmp_path):
    data = fixture(tmp_path)
    cases_before = copy.deepcopy(data[2])
    contexts, contract = load(data)
    assert len(contexts) == len(contract["cases"]) == 30
    assert contract["summary"] == {"machine_reviewed_sufficient_candidate": 1, "pending": 29}
    assert data[2] == cases_before
    assert all(r["acceptance_status"] == "pending_independent_review" for r in contract["cases"])


@pytest.mark.parametrize("field,value", [("split", "dev"), ("root_freeze_id", "0"*64), ("catalog_sha256", "0"*64), ("cases_sha256", "0"*64), ("sealed_manifest_sha256", "0"*64), ("sealed_cases_file_sha256", "0"*64), ("sealed_contexts_file_sha256", "0"*64), ("human_expert_approval", True), ("questions_modified", 1), ("method", "expert_review")])
def test_resealed_contract_cannot_bypass_global_provenance(tmp_path, field, value):
    data = fixture(tmp_path)
    data[4][field] = value
    reseal(data)
    with pytest.raises(ValueError):
        load(data)


@pytest.mark.parametrize("field,value", [("candidate_case_sha256", "0"*64), ("case_sha256", "0"*64), ("acceptance_status", "approved"), ("human_expert_approval", True), ("sufficiency", "approved"), ("reviewer", "offline_holdout_binding_auditor_v1"), ("query_relative_review_sha256", None)])
def test_resealed_row_cannot_bypass_case_or_machine_authority(tmp_path, field, value):
    data = fixture(tmp_path)
    data[4]["cases"][0][field] = value
    reseal(data)
    with pytest.raises(ValueError):
        load(data)


@pytest.mark.parametrize("field,value", [("source_hash", "b"*64), ("relative_path", "other.pdf"), ("location_id", "other"), ("block_id", "other"), ("page", 3), ("chm_member", "/other.htm"), ("quote", "Missing text"), ("quote_sha256", "0"*64)])
def test_resealed_claim_requires_exact_frozen_evidence(tmp_path, field, value):
    data = fixture(tmp_path)
    data[4]["cases"][0]["claim_slots"][0]["alternatives"][0]["evidence"][0][field] = value
    reseal(data)
    with pytest.raises(ValueError):
        load(data)


def test_duplicate_and_missing_reference_rows_rejected(tmp_path):
    data = fixture(tmp_path)
    data[4]["cases"][-1] = copy.deepcopy(data[4]["cases"][0])
    reseal(data)
    with pytest.raises(ValueError, match="coverage"):
        load(data)


def test_context_fingerprint_and_artifact_file_drift_rejected(tmp_path):
    data = fixture(tmp_path)
    data[3]["synthetic-000"]["records"][0]["text"] += " modified"
    reseal(data)
    with pytest.raises(ValueError, match="fingerprint"):
        load(data)
    path = data[1]/"holdout_candidate_contexts.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="file SHA"):
        load(data)


def test_current_sealed_files_cannot_change(tmp_path):
    data = fixture(tmp_path)
    path = data[0]/"holdout_cases.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="sealed"):
        load(data)


def test_unreviewed_claim_alternative_and_omitted_source_are_rejected(tmp_path):
    data = fixture(tmp_path)
    slot = data[4]["cases"][0]["claim_slots"][0]
    alternative = copy.deepcopy(slot["alternatives"][0])
    alternative["evidence"][0]["block_id"] = "not-frozen"
    slot["alternatives"].append(alternative)
    reseal(data)
    with pytest.raises(ValueError, match="binding"):
        load(data)


def test_import_performs_no_filesystem_reads(monkeypatch):
    import importlib

    import app.eval.holdout_reference_freeze as module
    def fail(*args, **kwargs):
        raise AssertionError("Import must not read sealed content")
    monkeypatch.setattr(module.Path, "read_text", fail)
    monkeypatch.setattr(module.Path, "read_bytes", fail)
    importlib.reload(module)
