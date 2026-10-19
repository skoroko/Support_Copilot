"""Load an explicitly unlocked, immutable machine reference candidate.

Importing this module performs no filesystem access, provider setup or API calls.
The caller must first unlock and load the original sealed holdout suite.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object identity")
        result[key] = value
    return result


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def _is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _bound(item, record):
    source = record["source"]
    block_ids = set(record["block_ids"]) | set(source.get("block_ids", []))
    if source.get("block_id"):
        block_ids.add(source["block_id"])
    return (isinstance(item.get("quote"), str) and bool(item["quote"])
            and item["quote"] in record["text"]
            and item.get("quote_sha256") == _digest(item["quote"])
            and item.get("relative_path") == source.get("relative_path")
            and item.get("source_hash") == source.get("file_sha256", source.get("source_hash"))
            and item.get("location_id") == source.get("location_id")
            and item.get("page") == source.get("physical_page", source.get("page"))
            and item.get("chm_member") == source.get("chm_member")
            and item.get("block_id") in block_ids)


def load_holdout_reference_freeze(directory: Path, suite: Path, original_cases: list[dict], *, candidate_identity_sha256: str, catalog_sha256: str) -> tuple[dict, dict]:
    """Return (contexts, contract), without changing original cases or authority.

    Mechanical binding establishes provenance only. Machine sufficient candidates
    remain pending independent reference acceptance; pending rows remain pending.
    """
    directory, suite = Path(directory), Path(suite)
    if not _is_sha(candidate_identity_sha256) or not _is_sha(catalog_sha256):
        raise ValueError("Explicit frozen candidate and catalog SHA256 required")
    audit_path = directory / "manifest.json"
    audit_file_sha = _file_sha(audit_path)
    audit = _read(audit_path)
    contract_path = directory / "holdout_binding_contract.json"
    contexts_path = directory / "holdout_candidate_contexts.json"
    paths = {"holdout_binding_contract.json": contract_path, "holdout_candidate_contexts.json": contexts_path}
    if set(audit.get("artifacts", {})) != set(paths):
        raise ValueError("Exact holdout reference artifact inventory required")
    for name, path in paths.items():
        if audit["artifacts"][name] != _file_sha(path):
            raise ValueError("Holdout reference artifact file SHA mismatch")
    contract, contexts = _read(contract_path), _read(contexts_path)
    for metadata in (audit, contract):
        if (metadata.get("split") != "holdout"
                or metadata.get("root_freeze_id") != candidate_identity_sha256
                or metadata.get("candidate_identity_sha256", candidate_identity_sha256) != candidate_identity_sha256
                or metadata.get("catalog_sha256") != catalog_sha256):
            raise ValueError("Holdout split/candidate/catalog freeze mismatch")
        if metadata.get("method") != "machine_source_review" or metadata.get("human_expert_approval") is not False:
            raise ValueError("Explicit machine-only reference authority required")
        if metadata.get("questions_modified") != 0:
            raise ValueError("Original questions cannot change")
    sealed = _read(suite / "manifest.json")
    sealed_manifest_sha = _file_sha(suite / "manifest.json")
    sealed_file_shas = {name: _file_sha(suite / name) for name in ("holdout_cases.json", "holdout_contexts.json")}
    if sealed.get("split_counts", {}).get("holdout") != 30 or len(original_cases) != 30:
        raise ValueError("Complete sealed30-case holdout required")
    original_hash = _digest(original_cases)
    for metadata in (audit, contract):
        if metadata.get("sealed_manifest_sha256") != _file_sha(suite / "manifest.json") or metadata.get("cases_sha256") != original_hash:
            raise ValueError("Original suite/global case binding mismatch")
        for name, key in (("holdout_cases.json", "sealed_cases_file_sha256"), ("holdout_contexts.json", "sealed_contexts_file_sha256")):
            actual = _file_sha(suite / name)
            if metadata.get(key) != actual or sealed.get("files_sha256", {}).get(name) != actual:
                raise ValueError("Original sealed case/context file changed")
    if _digest(_read(suite / "holdout_cases.json")) != original_hash:
        raise ValueError("Caller cases differ from sealed originals")
    by_case = {case["id"]: case for case in original_cases}
    rows = contract.get("cases", [])
    by_reference = {row["case_id"]: row for row in rows}
    if (len(by_case) != 30 or len(rows) != 30 or len(by_reference) != 30
            or set(by_reference) != set(by_case) or set(contexts) != set(by_case)):
        raise ValueError("Duplicate or incomplete holdout reference/context coverage")
    counts = {}
    for cid, case in by_case.items():
        if case.get("split") != "holdout" or case.get("reference_status") != "pending_review":
            raise ValueError("Original holdout scope or pending authority changed")
        row, frozen = by_reference[cid], contexts[cid]
        if row.get("candidate_case_sha256") != _digest(case) or row.get("case_sha256") != _digest(case):
            raise ValueError("Per-case reference binding mismatch")
        if (row.get("method") != "machine_source_review" or row.get("human_expert_approval") is not False
                or row.get("acceptance_status") != "pending_independent_review"):
            raise ValueError("Reference cannot promote itself to accepted/human authority")
        sufficiency = row.get("sufficiency")
        if sufficiency not in {"pending", "machine_reviewed_sufficient_candidate"}:
            raise ValueError("Unknown or promoted reference sufficiency")
        counts[sufficiency] = counts.get(sufficiency, 0) + 1
        records = frozen.get("records")
        if not isinstance(records, list) or frozen.get("fingerprint") != _digest(records):
            raise ValueError("Frozen holdout context fingerprint mismatch")
        allowed_sources = {a["file_sha256"] for a in case["required_evidence_slots"]}
        allowed_paths = {}
        for anchor in case["required_evidence_slots"]:
            if anchor.get("relative_path"):
                allowed_paths.setdefault(anchor["file_sha256"], set()).add(anchor["relative_path"])
        for record in records:
            source = record.get("source", {})
            ids = record.get("block_ids")
            if (not isinstance(record.get("text"), str) or not isinstance(ids, list)
                    or not all(isinstance(bid, str) and bid for bid in ids) or len(ids) != len(set(ids))
                    or source.get("file_sha256", source.get("source_hash")) not in allowed_sources
                    or not source.get("location_id") or not source.get("relative_path")):
                raise ValueError("Frozen context source/block provenance invalid")
            source_hash = source.get("file_sha256", source.get("source_hash"))
            if source_hash in allowed_paths and source["relative_path"] not in allowed_paths[source_hash]:
                raise ValueError("Frozen context changes original source path")
        if sufficiency == "pending":
            continue
        if (not _is_sha(audit.get("machine_review_file_sha256"))
                or not _is_sha(row.get("query_relative_review_sha256"))
                or not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip()
                or row["reviewer"] == "offline_holdout_binding_auditor_v1"
                or not isinstance(row.get("reference_answer"), str) or not row["reference_answer"].strip()):
            raise ValueError("Sufficient candidate requires separate query-relative machine review provenance")
        slots = row.get("claim_slots", [])
        if not slots and case.get("expected_behavior") not in {"abstain", "clarify"}:
            raise ValueError("Sufficient technical reference lacks required claim slots")
        seen_slots, covered = set(), set()
        for slot in slots:
            if not slot.get("slot_id") or slot["slot_id"] in seen_slots or not slot.get("claim") or not slot.get("alternatives"):
                raise ValueError("Duplicate/empty required claim slot")
            seen_slots.add(slot["slot_id"])
            for alternative in slot["alternatives"]:
                evidence = alternative.get("evidence", [])
                if not evidence:
                    raise ValueError("Empty required evidence conjunction")
                for item in evidence:
                    if not any(_bound(item, record) for record in records):
                        raise ValueError("Required claim lacks exact quote/source/location/block context binding")
                    covered.add(item["source_hash"])
        if covered != allowed_sources:
            raise ValueError("Sufficient reference omits required original source")
    if counts != contract.get("summary") or counts != audit.get("summary"):
        raise ValueError("Reference sufficiency summary differs from actual rows")
    # Catch concurrent edits after parsing, before returning trusted frozen material.
    if any(audit["artifacts"][name] != _file_sha(path) for name, path in paths.items()):
        raise ValueError("Reference artifacts changed during loading")
    if (_file_sha(audit_path) != audit_file_sha or _file_sha(suite / "manifest.json") != sealed_manifest_sha
            or any(_file_sha(suite/name) != expected for name, expected in sealed_file_shas.items())):
        raise ValueError("Sealed suite or reference manifest changed during loading")
    return contexts, contract
