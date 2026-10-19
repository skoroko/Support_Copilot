"""Provider-free checks of literal review provenance and fail-closed approval."""
from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest
from pypdf import PdfReader, PdfWriter

from app.eval.acceptance_budget import BudgetLedger
from app.eval.visual_literal_review import (
    bound_candidates,
    prepare_occurrence_crops,
    review_batches,
    review_candidates,
    review_request,
    reviewed_records,
)
from app.services.corpus import CorpusCatalog
from app.services.ingestion import _load_one_file
from app.services.visual_recovery import (
    build_visual_inventory,
    render_unique_visual_batch,
)
from tests.unit.test_corpus import write_pdf


@pytest.fixture
def review_fixture(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = data / "manual.pdf"
    write_pdf(source)
    original = PdfReader(source)
    writer = PdfWriter()
    writer.append(original)
    writer.append(original)
    with source.open("wb") as stream:
        writer.write(stream)
    catalog = CorpusCatalog.from_documents([source], _load_one_file(source, data_root=data), data_root=data)
    catalog_path = tmp_path / "catalog.json"
    catalog.persist(catalog_path)
    inventory = build_visual_inventory(catalog_path, data)
    manifest = tmp_path / "batch" / "batch_manifest.json"
    visual = render_unique_visual_batch(inventory, data, manifest.parent, scale=1.)
    assert visual["prepared_occurrences"] == 2
    group = visual["groups"][0]
    rows = [{**o, "group_id": group["group_id"], "text": "CANDIDATE_NOT_IN_PROMPT = 15 kPa", "kind": "formula",
             "approval_status": "pending", "crop_path": group["full_crop_path"],
             "crop_sha256": group["full_crop_sha256"]} for o in group["occurrences"]]
    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps({"records": rows}), encoding="utf-8")
    return data, manifest, visual, rows, pending


def response_for(rows):
    return {"reviews": [{"region_id": r["region_id"], "literal_reading": r["text"], "kind": r["kind"],
                         "all_visible_text_read": True, "uncertainty": "none", "uncertain_tokens": [], "warnings": []}
                        for r in rows]}


def prepared(review_fixture, tmp_path):
    data, manifest, inventory, rows, _ = review_fixture
    return prepare_occurrence_crops(rows, inventory, manifest_path=manifest, data_root=data,
                                    output_dir=tmp_path / "review_crops")


def test_each_duplicate_supplies_its_own_exact_crop_and_candidate_is_blind(review_fixture, tmp_path):
    rows = prepared(review_fixture, tmp_path)
    assert rows[0]["review_crop_path"] != rows[1]["review_crop_path"]
    assert all(r["reviewed_occurrence_region_id"] == r["region_id"] for r in rows)
    request = review_request(rows)
    labels = [r["text"] for r in request["input"][0]["content"] if r["type"] == "input_text"]
    assert len(labels) == 2 and all("CANDIDATE_NOT_IN_PROMPT" not in label for label in labels)
    assert sum(r["type"] == "input_image" for r in request["input"][0]["content"]) == 2
    rows[1]["reviewed_occurrence_region_id"] = rows[0]["region_id"]
    with pytest.raises(ValueError, match="binding changed"):
        review_request(rows)


@pytest.mark.parametrize("field,value", [("uncertainty", "partial"), ("uncertain_tokens", ["1/l"]),
                                         ("all_visible_text_read", False), ("warnings", ["clipped"]),
                                         ("literal_reading", "CANDIDATE_NOT_IN_PROMPT = 16 kPa"),
                                         ("kind", "definition")])
def test_uncertain_symbols_numbers_or_layout_never_get_approved(review_fixture, tmp_path, field, value):
    rows = prepared(review_fixture, tmp_path)
    payload = response_for(rows)
    payload["reviews"][1][field] = value
    decisions, approvals = reviewed_records(payload, rows, batch_id="fixture", reviewed_at="2026-10-16Z")
    assert len(approvals) == 1 and decisions[1]["decision"] == "pending_review"
    assert approvals[0]["semantic_verified"] is False and approvals[0]["human_expert_reviewed"] is False
    assert approvals[0]["same_model_independent_pass"] is True


@pytest.mark.parametrize("corruption", ["omit", "duplicate", "invented_region"])
def test_malformed_review_cannot_salvage_partial_approvals(review_fixture, tmp_path, corruption):
    rows = prepared(review_fixture, tmp_path)
    payload = response_for(rows)
    if corruption == "omit":
        payload["reviews"].pop()
    elif corruption == "duplicate":
        payload["reviews"][1] = payload["reviews"][0]
    else:
        payload["reviews"][1]["region_id"] = "invented"
    with pytest.raises(ValueError):
        reviewed_records(payload, rows, batch_id="fixture", reviewed_at="2026-10-16Z")


def test_wrong_source_identity_is_rejected_before_review(review_fixture):
    _, _, inventory, rows, pending = review_fixture
    rows[1]["version"] = "unseen"
    pending.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="exact source"):
        bound_candidates(pending, inventory)


def test_completed_review_resumes_without_hidden_replay(review_fixture, tmp_path):
    data, manifest, _, rows, pending = review_fixture
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="completed", output_text=json.dumps(response_for(rows)),
                               usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 100, "output_tokens": 100}))
    client = SimpleNamespace(base_url="https://api.openai.com/v1", max_retries=0,
                             responses=SimpleNamespace(create=create))
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd=10)
    kwargs = {"manifest_path": manifest, "candidates_path": pending, "data_root": data,
              "output_dir": tmp_path / "review", "ledger": ledger, "client": client}
    first = review_candidates(**kwargs)
    repeated = review_candidates(**kwargs)
    assert first["approved_literal_occurrences"] == repeated["approved_literal_occurrences"] == 2
    assert len(calls) == ledger.summary()["calls"] == 1
    assert calls[0]["service_tier"] == "default"
    assert len(json.loads((kwargs["output_dir"] / "approved_literal_transcriptions.json").read_text())) == 2


def test_unknown_provider_call_keeps_reserve_and_no_approval_or_retry(review_fixture, tmp_path):
    data, manifest, _, _, pending = review_fixture
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("fixture unknown acknowledgement")
    client = SimpleNamespace(base_url="https://api.openai.com/v1", max_retries=0,
                             responses=SimpleNamespace(create=create))
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd=10)
    kwargs = {"manifest_path": manifest, "candidates_path": pending, "data_root": data,
              "output_dir": tmp_path / "review", "ledger": ledger, "client": client}
    assert review_candidates(**kwargs)["failed_or_unknown_batches"] == 1
    assert review_candidates(**kwargs)["approved_literal_occurrences"] == 0
    assert len(calls) == 1 and ledger.summary()["unknown_cost_calls"] == 1
    assert ledger.summary()["accounted_upper_usd"] > 0


def test_original_source_change_never_reaches_provider(review_fixture, tmp_path):
    data, manifest, inventory, rows, _ = review_fixture
    (data / "manual.pdf").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source hash changed"):
        prepare_occurrence_crops(rows, inventory, manifest_path=manifest, data_root=data,
                                 output_dir=tmp_path / "crops")


def test_raw_response_is_durable_before_lost_settlement_ack(review_fixture, tmp_path, monkeypatch):
    data, manifest, _, rows, pending = review_fixture
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id="fixture-response-id", status="completed", output_text=json.dumps(response_for(rows)),
                               usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 100, "output_tokens": 100}))
    client = SimpleNamespace(base_url="https://api.openai.com/v1", max_retries=0,
                             responses=SimpleNamespace(create=create))
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd=10)
    output = tmp_path / "review"
    def lost_ack(*args, **kwargs):
        # The wrapper must durably save the provider result before even trying
        # settlement, rather than only saving it in the outer exception handler.
        with sqlite3.connect(output / "review_attempts.sqlite3") as db:
            saved = json.loads(db.execute("SELECT result FROM attempts").fetchone()[0])
        assert saved["state"] == "response_received_pending_parse"
        assert saved["response"]["usage"]["input_tokens"] == 100
        assert saved["response"]["output_text"] == json.dumps(response_for(rows))
        assert saved["provider_request_started"] is True
        raise RuntimeError("fixture lost settlement acknowledgement")
    monkeypatch.setattr(ledger, "settle", lost_ack)
    kwargs = {"manifest_path": manifest, "candidates_path": pending, "data_root": data,
              "output_dir": output, "ledger": ledger, "client": client}
    result = review_candidates(**kwargs)
    assert result["approved_literal_occurrences"] == 0 and result["failed_or_unknown_batches"] == 1
    assert review_candidates(**kwargs)["approved_literal_occurrences"] == 0
    assert len(calls) == 1 and ledger.summary()["unknown_cost_calls"] == 1
    with sqlite3.connect(output / "review_attempts.sqlite3") as db:
        saved = json.loads(db.execute("SELECT result FROM attempts").fetchone()[0])
    assert saved["response"]["id"] == "fixture-response-id" and saved["error_type"] == "RuntimeError"


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_first_failed_or_interrupted_batch_stops_and_exports(review_fixture, tmp_path, monkeypatch, failure):
    from app.eval import visual_literal_review

    # Force the two exact occurrences into two real batches, not one response.
    monkeypatch.setattr(visual_literal_review, "MAX_REVIEW_OCCURRENCES", 1)
    data, manifest, _, _, pending = review_fixture
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        raise failure("fixture interrupted provider")
    client = SimpleNamespace(base_url="https://api.openai.com/v1", max_retries=0,
                             responses=SimpleNamespace(create=create))
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd=10)
    output = tmp_path / "review"
    result = review_candidates(manifest_path=manifest, candidates_path=pending, data_root=data,
                               output_dir=output, ledger=ledger, client=client)
    assert result["attempted_batches"] == len(calls) == 1
    assert result["failed_or_unknown_batches"] == 1 and ledger.summary()["unknown_cost_calls"] == 1
    assert json.loads((output / "approved_literal_transcriptions.json").read_text()) == []
    assert json.loads((output / "literal_review_decisions.json").read_text())["records"] == []


def test_batches_bound_each_occurrence_without_losing_duplicate_contexts():
    rows = [{"group_id": "repeated", "region_id": str(i)} for i in range(40)]
    rows.extend({"group_id": f"group-{i}", "region_id": f"unique-{i}"} for i in range(20))
    batches = review_batches(rows, 16)
    assert all(len(batch) <= 32 and len({row["group_id"] for row in batch}) <= 16 for batch in batches)
    assert [row for batch in batches for row in batch] == rows
