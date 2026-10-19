from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from app.eval.corpus_suite import (
    CatalogLexicalSmoke,
    ExtractionVerifier,
    bind_reviews,
    digest,
    evaluate,
    evidence_scores,
    file_sha,
    infer_group,
    load_suite,
    raw_row_sha,
    read_json,
    report,
    reviewed_cases,
    summarize,
    validate_cases,
    write_new,
)

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "tests/eval/corpus_suite"


@pytest.fixture
def bundle():
    manifest, cases, contexts = load_suite(SUITE)
    return manifest, cases, contexts, read_json(SUITE / "release_policy.json")


def test_sealed_suite_counts_and_pending_references(bundle):
    manifest, dev, _, _ = bundle
    _, holdout, _ = load_suite(SUITE, "holdout", unlock_holdout=True)
    assert len(dev) == 90 and len(holdout) == 30
    assert manifest["case_count"] == 120
    assert {c["reference_status"] for c in dev + holdout} == {"pending_review"}
    assert all(c["reference"] is None for c in dev + holdout)
    records = lambda cs: {a["document_record_id"] for c in cs for a in c["required_evidence_slots"]}
    assert not records(dev) & records(holdout)
    topics = lambda cs: {t for c in cs for t in c["topic_components"]}
    assert not topics(dev) & topics(holdout)
    assert all("not_expert_certified" in c["topic_split_assurance"] for c in dev + holdout)
    assert len({c["family_id"] for c in dev + holdout}) == 120
    assert len({a["relative_path"] for c in dev + holdout for a in c["required_evidence_slots"]
                if a["relative_path"].endswith(".pdf")}) == 9


def test_holdout_is_locked_and_not_read_by_default(monkeypatch, bundle):
    with pytest.raises(ValueError, match="Holdout"):
        load_suite(SUITE, "holdout")
    from app.eval import corpus_suite
    original = corpus_suite.read_json
    def guarded(path):
        assert path.name not in {"holdout_cases.json", "holdout_contexts.json"}
        return original(path)
    monkeypatch.setattr(corpus_suite, "read_json", guarded)
    load_suite(SUITE)


def test_required_multisource_slots_are_real_files(bundle):
    _, cases, _, _ = bundle
    for case in cases:
        anchors = case["required_evidence_slots"]
        assert infer_group(anchors) == case["source_group"]
        if case["source_group"] == "multi_pdf":
            assert len({a["relative_path"] for a in anchors}) >= 2
            assert all(a["relative_path"].endswith(".pdf") for a in anchors)
        if case["source_group"] == "mixed":
            assert any(a["relative_path"].endswith(".pdf") for a in anchors)
            assert any(not a["relative_path"].endswith(".pdf") for a in anchors)


def test_legacy_hashes_match_unchanged_worktree():
    legacy = read_json(SUITE / "legacy_manifest.json")
    assert legacy["counts"] == {"golden": 34, "supplemental_turns": 9, "expanded_modes": 5}
    for name, sha in legacy["files_sha256"].items():
        assert file_sha(Path(legacy["source_root"]) / name) == sha


def test_bad_quote_or_group_rejected(bundle):
    _, cases, _, _ = bundle
    bad = deepcopy(cases)
    bad[0]["required_evidence_slots"][0]["exact_quote"] += "fake"
    with pytest.raises(ValueError, match="Quote"):
        validate_cases(bad, "dev")
    bad = deepcopy(cases)
    bad[0]["source_group"] = "no_answer"
    with pytest.raises(ValueError, match="Source group"):
        validate_cases(bad, "dev")


def test_dry_run_constructs_no_provider_and_is_blocked(bundle):
    manifest, cases, contexts, policy = bundle
    run = evaluate(cases, contexts, split="dev", manifest=manifest, policy=policy)
    assert len(run["attempts"]) == 360
    assert all(r["status"] == "not_run" for r in run["attempts"])
    assert run["summary"]["release_allowed"] is False
    assert run["summary"]["overall"]["timing_all_attempts"]["p50_ms"] is None
    assert run["summary"]["overall"]["telegram_delivery_latency_ms"] == {"denominator": 0, "p50_ms": None, "p95_ms": None}


def test_incomplete_generation_preserves_partial_output_and_usage_as_failed(bundle):
    manifest, cases, contexts, policy = bundle
    partial = {"answer": "partial", "usage": {"input_tokens": 7, "output_tokens": 3},
               "response_metadata": {"status": "incomplete", "finish_reason": "length"}}
    def adapter(*args):
        error = RuntimeError("incomplete fixture")
        error.evaluation_output = partial
        raise error
    run = evaluate(cases[:1], contexts, split="dev", manifest=manifest, policy=policy,
                   adapters={"generation": adapter})
    row = next(r for r in run["attempts"] if r["layer"] == "generation")
    assert row["status"] == "failed" and row["output"] == partial
    assert row["usage"] == partial["usage"]
    assert run["summary"]["release_allowed"] is False


def test_profiles_receive_identical_frozen_material_and_cannot_mutate_it(bundle):
    manifest, cases, contexts, policy = bundle
    selected = cases[:1]
    seen = []
    def adapter(case, frozen, profile):
        seen.append(frozen["fingerprint"])
        frozen["records"][0]["text"] = "mutated adapter copy"
        return {"answer": "fixture"}
    before = digest(contexts)
    run = evaluate(selected, contexts, split="dev", manifest=manifest, policy=policy,
                   profiles={"one": {}, "two": {"effort": "low"}}, adapters={"generation": adapter})
    assert len(seen) == 2 and seen[0] == seen[1]
    assert digest(contexts) == before
    assert len({r["context_sha256"] for r in run["attempts"]}) == 1


def test_retrieval_requires_exact_support_not_filename_only(bundle):
    _, cases, contexts, _ = bundle
    case = next(c for c in cases if c["source_group"] == "multi_pdf")
    frozen = contexts[case["id"]]
    assert evidence_scores(case, {"records": frozen["records"]}) == {"evidence_recall_at_10": 1., "all_slots_at_10": 1.}
    partial = evidence_scores(case, {"records": frozen["records"][:1]})
    assert partial == {"evidence_recall_at_10": .5, "all_slots_at_10": 0.}
    assert evidence_scores(case, {"records": [{"source": frozen["records"][0]["source"], "text": "unrelated"}]})["evidence_recall_at_10"] == 0


def test_failures_count_and_safe_errors_do_not_leak_messages(bundle):
    manifest, cases, contexts, policy = bundle
    def failed(*args):
        raise RuntimeError("credential-like material must never appear")
    run = evaluate(cases[:1], contexts, split="dev", manifest=manifest, policy=policy,
                   adapters={"generation": failed, "retrieval": failed})
    rows = [r for r in run["attempts"] if r["status"] == "failed"]
    assert len(rows) == 2
    assert all(r["error"] == {"type": "RuntimeError"} for r in rows)
    summary = summarize(rows)
    assert summary["correctness"] == {"value": 0, "denominator": 1}
    assert summary["grounding"]["value"] is None
    assert summary["timing_all_attempts"]["denominator"] == 2
    assert summary["combined_quality_sources_under_15s"]["value"] == 0


def review_for(run, row):
    return {"run_id": run["run_id"], "run_binding_sha256": run["binding_sha256"], "reviews": [{
        "attempt_id": row["attempt_id"], "raw_attempt_sha256": raw_row_sha(row), "method": "expert",
        "reviewer": "test-reviewer", "notes": "Fixture expert binding only, not real corpus review",
        "correctness": 1, "completeness": 1, "grounding": 1, "sources_valid": True, "critical_error": False}]}


def test_expert_review_requires_exact_result_binding(bundle):
    manifest, cases, contexts, policy = bundle
    run = evaluate(cases[:1], contexts, split="dev", manifest=manifest, policy=policy,
                   adapters={"generation": lambda *args: {"answer": "fixture", "sources": []}})
    row = next(r for r in run["attempts"] if r["layer"] == "generation")
    review = review_for(run, row)
    row["output"]["answer"] = "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        bind_reviews(run, review)
    review = review_for(run, row)
    review["reviews"][0]["method"] = "regex_proxy"
    with pytest.raises(ValueError, match="named expert"):
        bind_reviews(run, review)


def test_slow_correct_answer_fails_combined_gate(bundle):
    manifest, cases, contexts, policy = bundle
    run = evaluate(cases[:1], contexts, split="dev", manifest=manifest, policy=policy,
                   adapters={"generation": lambda *args: {"answer": "fixture"}})
    row = next(r for r in run["attempts"] if r["layer"] == "generation")
    row["latency_ms"] = 15001
    bind_reviews(run, review_for(run, row))
    assert summarize([row])["combined_quality_sources_under_15s"]["value"] == 0
    assert report(run, cases[:1], policy)["release_allowed"] is False


def test_reference_approval_needs_expert_case_binding(bundle):
    manifest, cases, _, _ = bundle
    ledger = {"suite_manifest_sha256": digest(manifest), "references": [{
        "case_id": cases[0]["id"], "case_sha256": digest(cases[0]), "method": "expert", "status": "approved",
        "reviewer": "fixture", "notes": "test only", "reference": "test only", "applicability_checked": True}]}
    effective = reviewed_cases(cases, ledger, digest(manifest))
    assert effective[0]["reference_status"] == "approved"
    assert cases[0]["reference_status"] == "pending_review"
    ledger["references"][0]["case_sha256"] = "bad"
    with pytest.raises(ValueError, match="fingerprint"):
        reviewed_cases(cases, ledger, digest(manifest))


def test_missing_visual_regions_are_extraction_failure(bundle, monkeypatch):
    _, cases, contexts, _ = bundle
    case = next(c for c in cases if c.get("extraction_modalities") == ["text", "visual_region"])
    verifier = object.__new__(ExtractionVerifier)
    verifier.integrity = verifier.docstore_integrity = True
    verifier.acquisition_ms = 0
    verifier.visual_catalog = None
    verifier.records = {a["document_record_id"]: {"text": a["exact_quote"]} for a in case["required_evidence_slots"]}
    verifier.hashes = {a["relative_path"]: a["file_sha256"] for a in case["required_evidence_slots"]}
    def texts(relative):
        anchor = next(a for a in case["required_evidence_slots"] if a["relative_path"] == relative)
        return [anchor["exact_quote"]] * anchor["physical_page"]
    monkeypatch.setattr(verifier, "texts", texts)
    result = verifier(case, contexts[case["id"]], {})
    assert result["extraction_pass"] is False
    assert result["anchor_checks"][0]["snapshot_quote_match"]
    assert result["anchor_checks"][0]["required_visual_regions_verified"] is False


def test_source_verification_uses_literal_dom_not_structured_table_labels(tmp_path):
    from app.services.corpus_html import parse_html

    raw = (b"<html><head><title>Manual</title></head><body><h1>Parameters</h1>"
           b"<table><tr><th>Value</th><th>Unit</th></tr><tr><td>10</td><td>MPa</td></tr></table>"
           b"<script>ignore this</script></body></html>")
    path = tmp_path / "manual.html"
    path.write_bytes(raw)
    verifier = object.__new__(ExtractionVerifier)
    verifier.cached, verifier.acquisition_ms, verifier.data_dir = {}, 0.0, tmp_path
    assert verifier.texts("manual.html") == ["Manual Parameters Value Unit 10 MPa"]
    assert "Column 1=" in parse_html(raw, "location")["text"]
    # The baseline proves source presence, not correctness of the transformed table.
    assert "Column 1=" not in verifier.texts("manual.html")[0]


def test_chm_source_baseline_reads_exact_members_and_propagates_failure(tmp_path, monkeypatch):
    import pylibmspack

    class Archive:
        def __init__(self, path):
            self.path = path

        def files(self, include_system=False):
            return [{"name": "/b.htm"}, {"name": "/a.htm"}, {"name": "/logo.png"}]

        def read(self, name, *, max_size):
            assert max_size == 16 * 1024 * 1024
            return {"/a.htm": b"<title>A</title><p>first</p>",
                    "/b.htm": b"<title>B</title><p>second</p>"}[name]

    monkeypatch.setattr(pylibmspack, "ChmArchive", Archive)
    (tmp_path / "manual.chm").write_bytes(b"archive fixture")
    verifier = object.__new__(ExtractionVerifier)
    verifier.cached, verifier.acquisition_ms, verifier.data_dir = {}, 0.0, tmp_path
    verifier.scratch_dir = tmp_path / "scratch"
    assert verifier.texts("manual.chm") == ["A first", "B second"]
    verifier.cached.clear()

    def unreadable(*args, **kwargs):
        raise OSError("unreadable member")

    monkeypatch.setattr(Archive, "read", unreadable)
    with pytest.raises(OSError, match="unreadable member"):
        verifier.texts("manual.chm")
    assert "manual.chm" not in verifier.cached


def test_run_artifacts_never_overwrite(tmp_path):
    path = tmp_path / "result.json"
    write_new(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_new(path, {"a": 2})


def test_offline_catalog_smoke_is_explicitly_not_production(tmp_path, bundle):
    manifest, cases, contexts, policy = bundle
    case = cases[0]
    anchor = case["required_evidence_slots"][0]
    catalog = {"sources": {"s": {"source_hash": anchor["file_sha256"], "relative_path": anchor["relative_path"]}},
               "locations": [{"location_id": "l", "page": anchor["physical_page"]}],
               "blocks": [{"source_id": "s", "location_id": "l", "block_id": "b", "kind": "paragraph",
                           "text": case["question"] + " " + anchor["exact_quote"]}]}
    path = tmp_path / "catalog.json"
    write_new(path, catalog)
    smoke = CatalogLexicalSmoke(path)
    result = smoke(case, contexts[case["id"]], {})
    assert result["network_calls"] == 0 and result["catalog_sha256"] == file_sha(path)
    assert result["records"][0]["source"]["block_id"] == "b"
    run = evaluate([case], contexts, split="dev", manifest=manifest, policy=policy, adapters={"retrieval": smoke})
    assert any("lexical catalog smoke" in reason for reason in run["summary"]["reasons"])


def test_e2e_uses_question_clock_not_adapter_history_overhead(bundle):
    manifest, cases, contexts, policy = bundle
    run = evaluate(cases[:1], contexts, split="dev", manifest=manifest, policy=policy,
                   adapters={"e2e": lambda *args: {"answer": "fixture", "total_latency_ms": 7,
                                                  "history_setup_ms": 17000}})
    row = next(r for r in run["attempts"] if r["layer"] == "e2e")
    assert row["latency_ms"] == 7 and row["history_setup_ms"] == 17000
    assert row["adapter_wall_ms"] is not None
    assert row["telegram_delivery_latency_ms"] is None


def test_http_completion_cannot_pass_delivered_answer_gate(bundle):
    manifest, cases, contexts, policy = bundle
    run = evaluate(cases[:1], contexts, split="dev", manifest=manifest, policy=policy,
                   adapters={"e2e": lambda *args: {"answer": "fixture", "total_latency_ms": 7}})
    row = next(r for r in run["attempts"] if r["layer"] == "e2e")
    bind_reviews(run, review_for(run, row))
    assert summarize([row])["combined_quality_sources_under_15s"]["passed"] == 0
    assert any("observed user delivery missing" in reason for reason in report(run, cases[:1], policy)["reasons"])
    row["telegram_delivery_latency_ms"] = 15001
    assert summarize([row])["combined_quality_sources_under_15s"]["passed"] == 0
    row["telegram_delivery_latency_ms"] = 12000
    assert summarize([row])["combined_quality_sources_under_15s"]["passed"] == 1


def test_incomplete_but_correct_answer_fails_combined_gate(bundle):
    manifest, cases, contexts, policy = bundle
    run = evaluate(cases[:1], contexts, split="dev", manifest=manifest, policy=policy,
                   adapters={"generation": lambda *args: {"answer": "fixture"}})
    row = next(r for r in run["attempts"] if r["layer"] == "generation")
    review = review_for(run, row)
    review["reviews"][0]["completeness"] = .5
    bind_reviews(run, review)
    assert summarize([row])["combined_quality_sources_under_15s"]["passed"] == 0
    assert summarize([row])["completeness"]["value"] == .5


@pytest.mark.parametrize("latency", [-1, float("nan"), True])
def test_invalid_delivery_latency_is_a_failed_attempt(bundle, latency):
    manifest, cases, contexts, policy = bundle
    run = evaluate(cases[:1], contexts, split="dev", manifest=manifest, policy=policy,
                   adapters={"e2e": lambda *args: {"answer": "fixture", "total_latency_ms": 7,
                                                  "telegram_delivery_latency_ms": latency}})
    assert next(r for r in run["attempts"] if r["layer"] == "e2e")["status"] == "failed"


@pytest.mark.parametrize("partial", [None, {"answer": "partial", "usage": {"output_tokens": 1}},
                                     {"total_latency_ms": float("nan"), "history_setup_ms": -1}])
def test_e2e_failure_keeps_separate_question_timing(bundle, partial):
    manifest, cases, contexts, policy = bundle
    def failing(*args):
        error = RuntimeError("do not serialize")
        error.evaluation_latency_ms = 16000
        error.evaluation_history_setup_ms = 12000
        if partial is not None:
            error.evaluation_output = partial
        raise error
    run = evaluate(cases[:1], contexts, split="dev", manifest=manifest, policy=policy, adapters={"e2e": failing})
    row = next(r for r in run["attempts"] if r["layer"] == "e2e")
    assert row["latency_ms"] == 16000 and row["history_setup_ms"] == 12000
    assert summarize([row])["timing_all_attempts"]["p95_ms"] == 16000


def test_cli_dry_run_and_live_permission_flags(tmp_path):
    script = ROOT / "scripts/evaluate_corpus_suite.py"
    destination = tmp_path / "dry.json"
    result = subprocess.run([sys.executable, str(script), "--output", str(destination)],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert read_json(destination)["summary"]["overall"]["not_run"] == 360
    denied = subprocess.run([sys.executable, str(script), "--live", "--output", str(tmp_path / "live.json")],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    assert denied.returncode != 0
    assert not (tmp_path / "live.json").exists()


def test_source_bound_reference_contract_and_mechanical_slots():
    from app.eval.corpus_suite import slot_block_grounding, validate_reference_contract
    case = {"id": "dev1", "question": "q"}
    catalog = {"sources": [{"source_id": "s", "source_hash": "hash", "relative_path": "m.pdf"}],
               "locations": [{"location_id": "l", "page": 2}],
               "blocks": [{"block_id": "b", "source_id": "s", "location_id": "l", "text": "Full useful evidence.", "kind": "paragraph"}]}
    evidence = {"relative_path": "m.pdf", "source_hash": "hash", "page": 2,
                "location_id": "l", "block_id": "b", "quote": "useful evidence", "quote_sha256": digest("useful evidence")}
    row = {"case_id": "dev1", "case_sha256": digest(case), "acceptance_status": "pending_independent_review",
           "claim_slots": [{"slot_id": "claim1", "claim": "useful claim", "alternatives": [evidence]}]}
    contract = {"method": "machine_source_review", "human_expert_approval": False,
                "dev_cases_sha256": digest([case]), "catalog_sha256": "catalog", "cases": [row]}
    validate_reference_contract(contract, [case], catalog, catalog_sha256="catalog")
    record = {"source": {"file_sha256": "hash", "location_id": "l", "block_id": "b"}, "text": "Full useful evidence."}
    assert slot_block_grounding(row, [record])["all_slots"] is True
    assert slot_block_grounding(row, [record])["semantic_success"] is None
    wrong = deepcopy(record); wrong["source"]["file_sha256"] = "other"
    assert slot_block_grounding(row, [wrong])["all_slots"] is False
    changed = deepcopy(contract); changed["cases"][0]["claim_slots"][0]["alternatives"][0]["quote"] = "invented"
    with pytest.raises(ValueError):
        validate_reference_contract(changed, [case], catalog, catalog_sha256="catalog")
    changed = deepcopy(contract); changed["cases"][0]["acceptance_status"] = "approved"
    with pytest.raises(ValueError):
        validate_reference_contract(changed, [case], catalog, catalog_sha256="catalog")


def test_blind_machine_packet_and_budget_cannot_self_approve():
    from app.eval.corpus_suite import blind_machine_review_packet, budget_reservation
    case = {"id": "dev1", "question": "q", "expected_behavior": "answer"}
    reference = {"case_sha256": digest(case), "reference_answer": "reference", "claim_slots": []}
    attempt = {"case_id": "dev1", "status": "completed", "profile": "hidden_model_label", "output": {"answer": "actual output"}}
    packet = blind_machine_review_packet(case, reference, attempt)
    assert packet["answer_data"] == "actual output" and packet["approval_authority"] == "none"
    assert "hidden_model_label" not in str(packet)
    assert packet["attempt_sha256"] == digest(attempt)
    with pytest.raises(ValueError):
        budget_reservation({"quota_usd": 10, "quota_authorized_by_root": False}, request_id="a", maximum_usd=.1)
    ledger = budget_reservation({"quota_usd": 1, "quota_authorized_by_root": True}, request_id="a", maximum_usd=.7)
    ledger["attempts"][0]["status"] = "failed"
    with pytest.raises(ValueError):
        budget_reservation(ledger, request_id="b", maximum_usd=.4)
    with pytest.raises(ValueError):
        budget_reservation(ledger, request_id="a", maximum_usd=.1)


def test_versioned_dev_candidate_preserves_original_freeze():
    from app.eval.corpus_suite import load_dev_candidate
    manifest, cases, _ = load_suite(SUITE)
    candidate = SUITE / "dev_v2_9_r3"
    audit, v2cases, contexts, contract = load_dev_candidate(candidate, manifest, cases)
    assert len(v2cases) == len(contexts) == len(contract["cases"]) == 90
    assert all(c["reference_status"] == "pending_review" for c in v2cases)
    assert audit["split"] == "dev"
    assert v2cases[14]["question"] != cases[14]["question"]
    assert all(contexts[c["id"]]["records"] for c in v2cases if c["source_group"] != "no_answer")


@pytest.mark.parametrize("candidate_name", ["dev_v2_6", "dev_v2_8", "dev_v2_9_r2"])
def test_historical_reference_freezes_with_stale_global_case_hash_are_rejected(candidate_name):
    from app.eval.corpus_suite import load_dev_candidate
    manifest, cases, _ = load_suite(SUITE)
    with pytest.raises(ValueError, match="contract/cases binding mismatch"):
        load_dev_candidate(SUITE / candidate_name, manifest, cases)


def test_curated_claim_conjunction_requires_all_source_blocks():
    from app.eval.corpus_suite import slot_block_grounding
    def evidence(block, quote):
        return {"source_hash": "s", "location_id": "l", "block_id": block, "quote": quote}
    row = {"claim_slots": [{"slot_id": "fact", "claim": "whole fact", "alternatives": [
        {"evidence": [evidence("a", "first"), evidence("b", "second")]}]}]}
    records = [{"source": {"file_sha256": "s", "location_id": "l", "block_id": b}, "text": q}
               for b, q in [("a", "first"), ("b", "second")]]
    assert slot_block_grounding(row, records)["all_slots"] is True
    assert slot_block_grounding(row, records[:1])["all_slots"] is False


def test_group_coverage_accepts_actual_chunk_block_ids_preserves_math_case():
    from app.eval.corpus_suite import slot_block_grounding
    row = {"claim_slots": [{"slot_id": "formula", "claim": "formula", "alternatives": [{"evidence": [
        {"source_hash": "s", "location_id": "l", "block_id": "a", "quote": "E = x"}]}]}]}
    record = {"source": {"source_hash": "s", "location_id": "l", "block_ids": ["a", "b"]}, "text": "E=x"}
    assert slot_block_grounding(row, [record])["all_slots"] is True
    record["text"] = "e=x"
    assert slot_block_grounding(row, [record])["all_slots"] is False


def test_visual_reference_requires_exact_review_provenance():
    from app.eval.corpus_suite import validate_reference_contract
    case = {"id": "visual", "question": "formula"}
    attrs = {"method": "reviewed_visual_transcription", "verified": True,
             "review_id": "literal-review", "reviewer": "independent-reviewer",
             "reviewed_at": "2026-10-16", "image_sha256": "image"}
    catalog = {"sources": [{"source_id": "s", "source_hash": "hash", "relative_path": "m.pdf"}],
               "locations": [{"location_id": "l", "page": 2}],
               "blocks": [{"block_id": "b", "source_id": "s", "location_id": "l",
                           "text": "E = x", "kind": "formula", "attributes": attrs}]}
    evidence = {"relative_path": "m.pdf", "source_hash": "hash", "page": 2, "location_id": "l",
                "block_id": "b", "quote": "E = x", "quote_sha256": digest("E = x"), "source_review": attrs}
    contract = {"method": "machine_source_review", "human_expert_approval": False,
                "dev_cases_sha256": digest([case]), "catalog_sha256": "catalog",
                "cases": [{"case_id": "visual", "case_sha256": digest(case),
                           "acceptance_status": "pending_independent_review", "claim_slots": [
                               {"claim": "formula", "alternatives": [{"evidence": [evidence]}]}]}]}
    validate_reference_contract(contract, [case], catalog, catalog_sha256="catalog")
    stale = deepcopy(contract)
    stale["cases"][0]["claim_slots"][0]["alternatives"][0]["evidence"][0]["source_review"]["image_sha256"] = "other"
    with pytest.raises(ValueError, match="source-bound review"):
        validate_reference_contract(stale, [case], catalog, catalog_sha256="catalog")


def test_evaluation_uses_bound_curated_slots_and_preserves_legacy_metric(bundle):
    manifest, cases, contexts, policy = bundle
    case = cases[0]
    def evidence(block, quote):
        return {"source_hash": "source", "location_id": "location", "block_id": block, "quote": quote}
    reference = {"case_id": case["id"], "candidate_case_sha256": digest(case), "claim_slots": [
        {"slot_id": "first", "claim": "first fact", "alternatives": [{"evidence": [evidence("a", "E = x")]}]},
        {"slot_id": "second", "claim": "second fact", "alternatives": [{"evidence": [evidence("b", "second fact")]}]}]}
    contract = {"cases": [reference]}
    records = [{"source": {"source_hash": "source", "location_id": "location", "block_ids": ["a"]}, "text": "E=x"}]
    run = evaluate([case], {case["id"]: contexts[case["id"]]}, split="dev", manifest=manifest,
                   adapters={"retrieval": lambda *args: {"records": records}}, policy=policy,
                   reference_contract=contract)
    row = next(r for r in run["attempts"] if r["layer"] == "retrieval")
    assert row["metrics"] == {"evidence_recall_at_10": .5, "all_slots_at_10": 0.}
    assert row["legacy_anchor_metrics"]["evidence_recall_at_10"] == 0.
    assert row["mechanical_slot_block_coverage"]["semantic_success"] is None
    assert run["machine_reference_contract_sha256"] == digest(contract)
    stale = deepcopy(contract)
    stale["cases"][0]["candidate_case_sha256"] = "stale"
    with pytest.raises(ValueError, match="does not bind"):
        evaluate([case], {case["id"]: contexts[case["id"]]}, split="dev", manifest=manifest,
                 adapters={"retrieval": lambda *args: pytest.fail("Adapter called before contract validation")},
                 policy=policy, reference_contract=stale)
