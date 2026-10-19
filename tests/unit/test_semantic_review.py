from __future__ import annotations

import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from app.eval.acceptance_budget import BudgetLedger
from app.eval.semantic_review import (
    blind_packet,
    parse_reviews,
    reference_block_reason,
    review_run,
)
from app.eval.visual_transcribe import canonical_hash


def fixtures(tmp_path):
    cases, contexts, refs, rows = [], {}, [], []
    for index in range(1, 4):
        case = {"id": f"case{index}", "question": "What does the parameter mean?", "history": [],
                "expected_behavior": "answer", "task_type": "definition", "product": "gts_nx"}
        source = {"relative_path": "manual.pdf", "file_sha256": "sourcehash", "physical_page": 1,
                  "location_id": f"location{index}", "product": "gts_nx", "version": "unknown"}
        record = {"source": source, "text": "Parameter x defines a bounded value under condition y.", "block_ids": [f"block{index}"]}
        contexts[case["id"]] = {"records": [record], "fingerprint": canonical_hash([record])}
        quote = "Parameter x defines a bounded value"
        refs.append({"case_id": case["id"], "candidate_case_sha256": canonical_hash(case),
            "acceptance_status": "pending_independent_review",
            "sufficiency": "pending" if index == 2 else "machine_reviewed_sufficient_candidate",
            "reference_answer": "x defines the value under y.", "limitations": [],
            "claim_slots": [{"slot_id": "definition", "claim": quote, "alternatives": [{"evidence": [{
                "relative_path": "manual.pdf", "source_hash": "sourcehash", "location_id": f"location{index}",
                "block_id": f"block{index}", "page": 1, "quote": quote, "quote_sha256": canonical_hash(quote)}]}]}]})
        cases.append(case)
        for repeat in range(2 if index == 1 else 1):
            rows.append({"attempt_id": f"SECRET_PROFILE_{repeat}:generation:{case['id']}:{repeat}",
                "case_id": case["id"], "profile": f"SECRET_PROFILE_{repeat}", "layer": "generation", "status": "completed",
                "context_sha256": contexts[case["id"]]["fingerprint"],
                "output": {"answer": "x is bounded under y [1].", "sources": [source], "model": "HIDDEN_MODEL"}})
    rows.extend([{**rows[0], "attempt_id": "failed", "status": "failed", "output": None},
                 {**rows[0], "attempt_id": "notrun", "status": "not_run", "output": None}])
    contract = {"method": "machine_source_review", "human_expert_approval": False,
                "dev_cases_sha256": canonical_hash(cases), "cases": refs}
    run = {"run_id": "run", "binding_sha256": "binding", "frozen_context_sha256": canonical_hash(contexts),
           "machine_reference_contract_sha256": canonical_hash(contract), "attempts": rows}
    paths = []
    for name, value in (("run", run), ("cases", cases), ("contexts", contexts), ("contract", contract)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    return paths, run, cases, contexts, refs


def model_payload(packet, *, sufficient=True):
    return {"reviews": [{"answer_id": answer["answer_id"], "correctness": .8 if sufficient else None,
        "completeness": .6 if sufficient else None, "grounding": .7 if sufficient else None,
        "critical_error": False if sufficient else None, "invalid_citations": [], "missing_required_claims": [],
        "contradictory_claims": [], "reference_sufficient": sufficient,
        "uncertainty": "none" if sufficient else "insufficient_reference",
        "evidence_rationale": [{"evidence_id": "E0001", "source_quote": "Parameter x defines a bounded value",
                                "answer_quote": "x is bounded", "rationale": "Definition is supported."}] if sufficient else [],
        "notes": "provisional"} for answer in packet["answers"]]}


class FakeClient:
    base_url = "https://api.openai.com/v1"
    max_retries = 0
    def __init__(self, *, failure=False, sufficient=True):
        self.calls, self.failure, self.sufficient = [], failure, sufficient
        self.responses = SimpleNamespace(create=self.create)
    def create(self, **request):
        self.calls.append(request)
        if self.failure:
            raise TimeoutError("private exception containing token")
        packet = json.loads(request["input"])
        usage = {"input_tokens": 100, "output_tokens": 100, "total_tokens": 200}
        return SimpleNamespace(id="response", status="completed", output_text=json.dumps(model_payload(packet, sufficient=self.sufficient)),
            usage=SimpleNamespace(model_dump=lambda: usage), output=[])


def call(paths, tmp_path, client, **kwargs):
    return review_run(run_path=paths[0], cases_path=paths[1], contexts_path=paths[2], contract_path=paths[3],
        output_dir=tmp_path / "review", ledger=BudgetLedger(tmp_path / "budget.sqlite3"), client=client, **kwargs)


def test_blind_group_preserves_actual_citations_and_full_evidence(tmp_path):
    _paths, run, cases, contexts, refs = fixtures(tmp_path)
    attempts = run["attempts"][:2]
    packet, mapping = blind_packet(cases[0], refs[0], contexts["case1"], attempts)
    encoded = json.dumps(packet)
    assert "SECRET_PROFILE" not in encoded and "HIDDEN_MODEL" not in encoded
    assert len(packet["evidence"]) == 1 and len(packet["answers"]) == 2
    assert packet["answers"][0]["actual_cited_sources"][0]["source"]["location_id"] == "location1"
    assert packet["answers"][0]["answer_data"] == attempts[0]["output"]["answer"]
    assert set(mapping) == {"A0001", "A0002"}
    assert packet["evidence"][0]["text"] == contexts["case1"]["records"][0]["text"]


def test_pending_or_unbound_reference_remains_blocked(tmp_path):
    _, _, cases, contexts, refs = fixtures(tmp_path)
    assert reference_block_reason(cases[1], refs[1], contexts["case2"]) == "reference_pending_or_insufficient"
    changed = copy.deepcopy(refs[0])
    changed["claim_slots"][0]["alternatives"][0]["evidence"][0]["source_hash"] = "different_scope"
    assert reference_block_reason(cases[0], changed, contexts["case1"]) == "required_claim_lacks_exact_bound_evidence"


@pytest.mark.parametrize("changed_input", ["contract", "cases"])
def test_changed_reference_inputs_are_rejected_before_provider(tmp_path, changed_input):
    paths, *_ = fixtures(tmp_path)
    position = 3 if changed_input == "contract" else 1
    value = json.loads(paths[position].read_text(encoding="utf-8"))
    if changed_input == "contract":
        value["cases"][0]["reference_answer"] = "Different reference answer."
    else:
        value[0]["question"] = "Different question?"
    paths[position].write_text(json.dumps(value), encoding="utf-8")
    client = FakeClient()
    with pytest.raises(ValueError, match="binding mismatch"):
        call(paths, tmp_path, client)
    assert client.calls == []


def test_unsupported_sol_reasoning_is_rejected_before_provider_or_spend(tmp_path):
    paths, *_ = fixtures(tmp_path)
    client = FakeClient()
    with pytest.raises(ValueError, match="configuration"):
        call(paths, tmp_path, client, effort="none")
    assert client.calls == []
    assert BudgetLedger(tmp_path / "budget.sqlite3").summary()["calls"] == 0


def test_judge_may_abstain_but_cannot_score_insufficient_reference(tmp_path):
    _, run, cases, contexts, refs = fixtures(tmp_path)
    packet, mapping = blind_packet(cases[0], refs[0], contexts["case1"], run["attempts"][:2])
    payload = model_payload(packet, sufficient=False)
    reviews = parse_reviews(payload, packet, mapping)
    assert all(row["status"] == "reference_blocked" and row["correctness"] is None for row in reviews)
    payload["reviews"][0]["correctness"] = 1
    with pytest.raises(ValueError, match="unscored"):
        parse_reviews(payload, packet, mapping)


@pytest.mark.parametrize("change", ["quote", "identity", "score", "missing_slot"])
def test_hallucinated_quotes_and_wrong_answer_binding_rejected(tmp_path, change):
    _, run, cases, contexts, refs = fixtures(tmp_path)
    packet, mapping = blind_packet(cases[0], refs[0], contexts["case1"], run["attempts"][:2])
    payload = model_payload(packet)
    first = payload["reviews"][0]
    if change == "quote":
        first["evidence_rationale"][0]["source_quote"] = "invented formula"
    elif change == "identity":
        first["answer_id"] = "SECRET_PROFILE_0"
    elif change == "score":
        first["grounding"] = 1.1
    else:
        first["missing_required_claims"] = ["invented_slot"]
    with pytest.raises(ValueError):
        parse_reviews(payload, packet, mapping)


def test_machine_summary_has_pending_failed_denominators_and_does_not_mutate_run(tmp_path):
    paths, run, _, _, _ = fixtures(tmp_path)
    original_bytes = paths[0].read_bytes()
    client = FakeClient()
    artifact = call(paths, tmp_path, client)
    assert len(client.calls) == 2  # repeated case1 answers share evidence/call; pending case2 makes no API call
    assert all(request["model"] == "gpt-6.1-sol" and request["max_output_tokens"] == 4096
               and request["service_tier"] == "default" for request in client.calls)
    result = artifact["summary"]
    assert result["planned_semantic_attempts"] == 6 and result["candidate_failed"] == 1 and result["not_run"] == 1
    assert result["machine_scored"] == 3 and result["reference_blocked"] == 1
    assert result["metrics"]["correctness"]["including_failed_denominator"] == 4
    assert result["human_expert_approval"] is False and result["formal_acceptance"] == "not_assessed"
    assert artifact["method"] != "expert" and artifact["approval_authority"] == "none"
    assert paths[0].read_bytes() == original_bytes
    assert not any("semantic_review" in row for row in run["attempts"])
    call(paths, tmp_path, client)
    assert len(client.calls) == 2


def test_unknown_request_keeps_reservation_and_resume_only_unattempted(tmp_path):
    paths, _, _, _, _ = fixtures(tmp_path)
    failed = FakeClient(failure=True)
    first = call(paths, tmp_path, failed)
    assert len(failed.calls) == 1 and first["budget"]["unknown_cost_calls"] == 1
    assert "private exception" not in json.dumps(first)
    resumed = FakeClient()
    second = call(paths, tmp_path, resumed)
    assert len(resumed.calls) == 1  # unknown case1 never repeated; unattempted case3 may run
    assert second["summary"]["machine_scored"] == 1
    with sqlite3.connect(tmp_path / "review" / "review_attempts.sqlite3") as db:
        results = [json.loads(row[0]) for row in db.execute("SELECT result FROM attempts ORDER BY rowid")]
    assert results[0]["state"] == "failed_or_unknown"
    assert results[1]["response"]["usage"]["total_tokens"] == 200


def test_invalid_paid_quotes_stay_unscored_without_replaying_or_blocking_next_case(tmp_path):
    paths, _, _, _, _ = fixtures(tmp_path)
    class InvalidFirstClient(FakeClient):
        def create(self, **request):
            response = super().create(**request)
            if len(self.calls) == 1:
                payload = json.loads(response.output_text)
                payload["reviews"][0]["evidence_rationale"][0]["source_quote"] = "invented quote"
                response.output_text = json.dumps(payload)
            return response
    client = InvalidFirstClient()
    artifact = call(paths, tmp_path, client)
    assert len(client.calls) == 2
    assert artifact["review_attempt_states"] == {"completed_review_rejected": 1, "machine_review_completed": 1}
    assert artifact["summary"]["machine_scored"] == 1
    assert artifact["budget"]["unknown_cost_calls"] == 0
    call(paths, tmp_path, client)
    assert len(client.calls) == 2


def test_raw_output_persists_before_settlement_failure(tmp_path, monkeypatch):
    paths, _, _, _, _ = fixtures(tmp_path)
    ledger, client = BudgetLedger(tmp_path / "budget.sqlite3"), FakeClient()
    monkeypatch.setattr(ledger, "settle", lambda *a, **k: (_ for _ in ()).throw(ValueError("secret")))
    review_run(run_path=paths[0], cases_path=paths[1], contexts_path=paths[2], contract_path=paths[3],
        output_dir=tmp_path / "review", ledger=ledger, client=client)
    with sqlite3.connect(tmp_path / "review" / "review_attempts.sqlite3") as db:
        result = json.loads(db.execute("SELECT result FROM attempts").fetchone()[0])
    assert result["response"]["output_text"] and result["response"]["usage"]["total_tokens"] == 200
    assert result["reviews"] == [] and result["state"] == "failed_or_unknown"


def test_stale_saved_context_rejected_before_any_provider_request(tmp_path):
    paths, run, _, _, _ = fixtures(tmp_path)
    run["attempts"][0]["context_sha256"] = "wrong_context"
    paths[0].write_text(json.dumps(run))
    client = FakeClient()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        call(paths, tmp_path, client)
    assert client.calls == []


def test_actual_cited_evidence_is_preserved_and_shared_without_metadata_bloat(tmp_path):
    _, run, cases, contexts, refs = fixtures(tmp_path)
    for row in run["attempts"][:2]:
        row["output"]["sources"][0].update(text="Actual retrieved chunk [exact].", citation_index=7,
            block_attributes={"glyphs": [{"text": "huge audit payload"}]})
    packet, _ = blind_packet(cases[0], refs[0], contexts["case1"], run["attempts"][:2])
    cited = [row for row in packet["evidence"] if row.get("origin") == "actual_cited_source"]
    assert len(cited) == 1 and cited[0]["text"] == "Actual retrieved chunk [exact]."
    identities = [row["actual_cited_sources"][0] for row in packet["answers"]]
    assert identities[0]["evidence_ref"] == identities[1]["evidence_ref"]
    assert identities[0]["source"]["citation_index"] == 7
    assert "glyphs" not in json.dumps(packet)
