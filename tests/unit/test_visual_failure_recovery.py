from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from app.eval.acceptance_budget import BudgetLedger
from app.eval.visual_failure_recovery import (
    build_recovery_plan,
    build_unknown_transport_plan,
    coverage_report,
    load_recovery_plan,
    load_unknown_transport_plan,
    write_recovery_plan,
)
from app.eval.visual_transcribe import (
    canonical_hash,
    safe_error_fields,
    sha256,
    transcribe_inventory,
)
from app.services.corpus import file_hash, source_identity


class APIConnectionError(ConnectionError):
    pass


def fixture(tmp_path, count=9):
    groups = []
    for number in range(count):
        crop = tmp_path / f"crop{number}.png"
        crop.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([number]))
        occurrence = {"region_id": f"region{number}", "source_hash": "sourcehash", "source_id": "sourceid",
            "relative_path": "manual.pdf", "location_id": "location", "page": 1, "bbox": [0, 0, 10, 10],
            "image_sha256": f"image{number}", "product": "civil", "version": "unknown"}
        groups.append({"group_id": f"group{number}", "image_sha256": f"image{number}",
            "representative_region_id": occurrence["region_id"], "occurrences": [occurrence],
            "full_crop_path": str(crop), "full_crop_sha256": sha256(crop.read_bytes())})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"complete": True, "groups": groups}))
    ledger = BudgetLedger(tmp_path / "ledger.sqlite3")
    return manifest, ledger


class Client:
    base_url, max_retries = "https://api.openai.com/v1", 0

    def __init__(self, mode="malformed"):
        self.mode, self.calls = mode, []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **request):
        self.calls.append(request)
        if self.mode == "unknown":
            raise APIConnectionError("never expose this")
        labels = [json.loads(row["text"]) for row in request["input"][0]["content"] if row["type"] == "input_text"]
        payload = {"groups": [{"group_id": group["group_id"], "classification": "formula", "occurrences": [
            {"region_id": row["region_id"], "kind": "formula", "literal_transcription": "x=1",
             "uncertainty": "none", "uncertain_tokens": [], "warnings": []} for row in group["occurrences"]]}
            for group in labels]}
        usage = {"input_tokens": 100, "output_tokens": 100, "total_tokens": 200}
        return SimpleNamespace(id="response", status="incomplete" if self.mode == "incomplete" else "completed",
            output_text="{" if self.mode == "malformed" else json.dumps(payload), output=[],
            incomplete_details=SimpleNamespace(model_dump=lambda **kwargs: {"reason": "max_output_tokens"})
                if self.mode == "incomplete" else None,
            usage=SimpleNamespace(model_dump=lambda: usage))


def parent(tmp_path, mode="malformed", count=9):
    manifest, ledger = fixture(tmp_path, count)
    journal = tmp_path / "parent" / "attempts.sqlite3"
    transcribe_inventory(manifest_path=manifest, output_dir=journal.parent, ledger=ledger, client=Client(mode),
                         batch_size=8, max_batches=1)
    with sqlite3.connect(journal) as db:
        attempt_id = db.execute("SELECT id FROM attempts").fetchone()[0]
    return manifest, ledger, journal, attempt_id


@pytest.mark.parametrize("mode", ["malformed", "incomplete"])
def test_exact_paid_failure_explicit_new_batches_and_resume_never_replays(tmp_path, mode):
    manifest, ledger, journal, attempt_id = parent(tmp_path, mode)
    plan = build_recovery_plan(manifest_path=manifest, ledger_path=ledger.path, attempts=[(journal, attempt_id)])
    assert len(plan["groups"]) == 8
    assert plan["parents"][0]["paid_budget_call_id"]
    plan_path = tmp_path / "recovery.json"
    write_recovery_plan(plan_path, plan)
    with pytest.raises(FileExistsError):
        write_recovery_plan(plan_path, plan)
    client = Client("success")
    kwargs = {"manifest_path": manifest, "output_dir": tmp_path / "recovery", "ledger": ledger,
              "client": client, "batch_size": 4, "failed_known_recovery_plan": plan_path}
    result = transcribe_inventory(**kwargs)
    assert result["attempted_now"] == 2 and len(client.calls) == 2
    assert transcribe_inventory(**kwargs)["attempted_now"] == 0
    records = json.loads((tmp_path / "recovery/pending_transcriptions.json").read_text())["records"]
    assert {row["region_id"] for row in records} == {f"region{n}" for n in range(8)}
    assert all(row["approval_status"] == "pending" and row["verified"] is False
               and row["failed_known_recovery"]["plan_sha256"] == plan["plan_sha256"] for row in records)
    coverage = coverage_report(manifest_path=manifest, journals=[journal, tmp_path / "recovery/attempts.sqlite3"])["summary"]
    assert coverage["failed_known_occurrences"] == 0 and coverage["pending_candidate_occurrences"] == 8
    assert coverage["unattempted_occurrences"] == 1 and coverage["source_recovery_verified"] is False


def test_unknown_provider_failure_cannot_create_recovery_plan(tmp_path):
    manifest, ledger, journal, attempt_id = parent(tmp_path, "unknown")
    with pytest.raises(ValueError, match="only persisted"):
        build_recovery_plan(manifest_path=manifest, ledger_path=ledger.path, attempts=[(journal, attempt_id)])
    summary = coverage_report(manifest_path=manifest, journals=[journal])["summary"]
    assert summary["unknown_occurrences"] == 8 and summary["failed_known_occurrences"] == 0


@pytest.mark.parametrize("mutation", ["reserved", "usage", "fingerprint", "duplicate"])
def test_known_response_requires_one_exact_settled_paid_call(tmp_path, mutation):
    manifest, ledger, journal, attempt_id = parent(tmp_path)
    with sqlite3.connect(ledger.path) as db:
        if mutation == "duplicate":
            db.execute("INSERT INTO calls SELECT 'second',kind,model,fingerprint,reserved,charged,status,usage,started FROM calls")
        elif mutation == "usage":
            db.execute("UPDATE calls SET usage=?", (json.dumps({"input_tokens": 999, "output_tokens": 1}),))
        elif mutation == "fingerprint":
            db.execute("UPDATE calls SET fingerprint='other'")
        else:
            db.execute("UPDATE calls SET status='reserved'")
    with pytest.raises(ValueError, match="completed paid"):
        build_recovery_plan(manifest_path=manifest, ledger_path=ledger.path, attempts=[(journal, attempt_id)])


@pytest.mark.parametrize("change", ["plan", "manifest", "parent", "same_output", "batch_size"])
def test_plan_source_parent_configuration_and_new_journal_are_bound(tmp_path, change):
    manifest, ledger, journal, attempt_id = parent(tmp_path)
    plan = build_recovery_plan(manifest_path=manifest, ledger_path=ledger.path, attempts=[(journal, attempt_id)])
    plan_path = tmp_path / "plan.json"
    if change == "plan":
        plan["groups"][0]["region_ids"] = ["other"]
        plan["plan_sha256"] = canonical_hash({key: value for key, value in plan.items() if key != "plan_sha256"})
    write_recovery_plan(plan_path, plan)
    if change == "manifest":
        value = json.loads(manifest.read_text())
        value["groups"][0]["occurrences"][0]["source_hash"] = "changed"
        manifest.write_text(json.dumps(value))
    if change == "parent":
        with sqlite3.connect(journal) as db:
            row = json.loads(db.execute("SELECT result FROM attempts").fetchone()[0])
            row["response"]["id"] = "changed"
            db.execute("UPDATE attempts SET result=?", (json.dumps(row),))
    with pytest.raises(ValueError):
        load_recovery_plan(plan_path, manifest_path=manifest, ledger_path=ledger.path,
            output_dir=journal.parent if change == "same_output" else tmp_path / "new",
            batch_size=8 if change == "batch_size" else 4)


def test_duplicate_parent_and_unknown_shaped_attempt_never_authorized(tmp_path):
    manifest, ledger, journal, attempt_id = parent(tmp_path)
    with pytest.raises(ValueError, match="duplicate parent"):
        build_recovery_plan(manifest_path=manifest, ledger_path=ledger.path,
                            attempts=[(journal, attempt_id), (journal, attempt_id)])
    with sqlite3.connect(journal) as db:
        result = json.loads(db.execute("SELECT result FROM attempts").fetchone()[0])
        result["region_ids"].append(result["region_ids"][0])
        db.execute("UPDATE attempts SET result=?", (json.dumps(result),))
    with pytest.raises(ValueError, match="unique parent"):
        build_recovery_plan(manifest_path=manifest, ledger_path=ledger.path, attempts=[(journal, attempt_id)])


def test_recovery_stops_on_first_failure_and_cannot_filter_plan(tmp_path):
    manifest, ledger, journal, attempt_id = parent(tmp_path)
    plan_path = tmp_path / "plan.json"
    write_recovery_plan(plan_path, build_recovery_plan(manifest_path=manifest, ledger_path=ledger.path,
                                                     attempts=[(journal, attempt_id)]))
    client = Client("unknown")
    kwargs = {"manifest_path": manifest, "ledger": ledger, "client": client, "batch_size": 4,
              "failed_known_recovery_plan": plan_path}
    result = transcribe_inventory(**kwargs, output_dir=tmp_path / "failed")
    assert result["attempted_now"] == 1 and len(client.calls) == 1
    with pytest.raises(ValueError, match="unfiltered"):
        transcribe_inventory(**kwargs, output_dir=tmp_path / "bad", shard_count=2)


def test_coverage_rejects_persisted_candidate_provenance_drift(tmp_path):
    manifest, _ledger, journal, _attempt_id = parent(tmp_path, "success")
    with sqlite3.connect(journal) as db:
        result = json.loads(db.execute("SELECT result FROM attempts").fetchone()[0])
        result["pending_records"][0]["source_hash"] = "different"
        db.execute("UPDATE attempts SET result=?", (json.dumps(result),))
    with pytest.raises(ValueError, match="candidate records mismatch"):
        coverage_report(manifest_path=manifest, journals=[journal])


def transport_parent(tmp_path, count=9):
    manifest, ledger = fixture(tmp_path, count)
    data = tmp_path / "data"
    data.mkdir()
    source = data / "manual.pdf"
    source.write_bytes(b"%PDF-1.4 original source fixture bytes")
    value = json.loads(manifest.read_text())
    for group in value["groups"]:
        for row in group["occurrences"]:
            row.update(source_id=source_identity(source, data)[0], source_hash=file_hash(source))
    manifest.write_text(json.dumps(value))
    output = tmp_path / "transport"
    transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger, client=Client("unknown"),
                         batch_size=8, max_batches=1)
    with sqlite3.connect(output / "attempts.sqlite3") as db:
        attempt_id = db.execute("SELECT id FROM attempts").fetchone()[0]
    return manifest, ledger, output / "attempts.sqlite3", attempt_id, data


def test_explicit_transport_new_call_retains_entire_parent_charge_and_exact_lineage(tmp_path):
    manifest, ledger, journal, attempt_id, data = transport_parent(tmp_path)
    plan = build_unknown_transport_plan(manifest_path=manifest, ledger_path=ledger.path, data_root=data,
        attempts=[(journal, attempt_id)], acknowledge_held_parent_reservations=True)
    assert plan["old_reservations_released"] is False and len(plan["groups"]) == 8
    parent_call = plan["parents"][0]["held_budget_call_id"]
    with sqlite3.connect(ledger.path) as db:
        before = db.execute("SELECT * FROM calls WHERE id=?", (parent_call,)).fetchone()
    path = tmp_path / "transport_plan.json"
    write_recovery_plan(path, plan)
    client = Client("success")
    kwargs = {"manifest_path": manifest, "ledger": ledger, "client": client, "output_dir": tmp_path / "new_calls",
        "batch_size": 4, "unknown_transport_new_attempt_plan": path,
        "acknowledge_held_parent_reservations": True}
    result = transcribe_inventory(**kwargs)
    assert result["attempted_now"] == 2 and len(client.calls) == 2
    assert transcribe_inventory(**kwargs)["attempted_now"] == 0
    with sqlite3.connect(ledger.path) as db:
        assert db.execute("SELECT * FROM calls WHERE id=?", (parent_call,)).fetchone() == before
    records = json.loads((tmp_path / "new_calls/pending_transcriptions.json").read_text())["records"]
    assert {row["region_id"] for row in records} == {f"region{i}" for i in range(8)}
    assert all(row["unknown_transport_new_attempt"]["parents"][0]["held_budget_call_id"] == parent_call
               and row["approval_status"] == "pending" and row["verified"] is False for row in records)
    summary = coverage_report(manifest_path=manifest, journals=[journal, tmp_path / "new_calls/attempts.sqlite3"])["summary"]
    assert summary["unknown_occurrences"] == 0 and summary["pending_candidate_occurrences"] == 8
    assert summary["unattempted_occurrences"] == 1


@pytest.mark.parametrize("change", ["unacknowledged", "reserved", "released", "usage", "duplicate", "http400", "in_flight", "received"])
def test_transport_new_call_rejects_unheld_unpriced_inflight_or_provider_rejection(tmp_path, change):
    manifest, ledger, journal, attempt_id, data = transport_parent(tmp_path)
    with sqlite3.connect(ledger.path) as db:
        if change == "reserved":
            db.execute("UPDATE calls SET status='reserved'")
        elif change == "released":
            db.execute("UPDATE calls SET charged=0")
        elif change == "usage":
            db.execute("UPDATE calls SET usage='{}'")
        elif change == "duplicate":
            db.execute("INSERT INTO calls SELECT 'duplicate',kind,model,fingerprint,reserved,charged,status,usage,started FROM calls")
    if change in {"http400", "in_flight", "received"}:
        with sqlite3.connect(journal) as db:
            row = json.loads(db.execute("SELECT result FROM attempts").fetchone()[0])
            if change == "http400":
                row.update(error_type="BadRequestError", error_fields={"http_status": 400})
            elif change == "in_flight":
                row.update(state="attempted_unknown", finished_at=None)
            else:
                row["response"] = {"status": "completed"}
            db.execute("UPDATE attempts SET state=?,result=?", (row["state"], json.dumps(row)))
    with pytest.raises(ValueError):
        build_unknown_transport_plan(manifest_path=manifest, ledger_path=ledger.path, data_root=data,
            attempts=[(journal, attempt_id)], acknowledge_held_parent_reservations=change != "unacknowledged")
    if change == "http400":
        summary = coverage_report(manifest_path=manifest, journals=[journal])["summary"]
        assert summary["provider_rejected_unpriced_occurrences"] == 8 and summary["unknown_occurrences"] == 0


@pytest.mark.parametrize("change", ["source", "crop", "ack", "same_journal", "batch", "plan"])
def test_transport_plan_revalidates_source_crop_parent_and_explicit_ack(tmp_path, change):
    manifest, ledger, journal, attempt_id, data = transport_parent(tmp_path)
    plan = build_unknown_transport_plan(manifest_path=manifest, ledger_path=ledger.path, data_root=data,
        attempts=[(journal, attempt_id)], acknowledge_held_parent_reservations=True)
    if change == "plan":
        plan["parents"][0]["held_charged_micros"] = 0
    path = tmp_path / "plan.json"
    write_recovery_plan(path, plan)
    if change == "source":
        (data / "manual.pdf").write_bytes(b"changed source")
    elif change == "crop":
        (tmp_path / "crop0.png").write_bytes(b"changed crop")
    with pytest.raises(ValueError):
        load_unknown_transport_plan(path, manifest_path=manifest, ledger_path=ledger.path,
            output_dir=journal.parent if change == "same_journal" else tmp_path / "new",
            batch_size=8 if change == "batch" else 4, crop="full",
            acknowledge_held_parent_reservations=change != "ack")


def test_transport_new_call_cannot_spend_old_hold_or_bypass_shared_cap(tmp_path):
    manifest, ledger, journal, attempt_id, data = transport_parent(tmp_path)
    plan = build_unknown_transport_plan(manifest_path=manifest, ledger_path=ledger.path, data_root=data,
        attempts=[(journal, attempt_id)], acknowledge_held_parent_reservations=True)
    path = tmp_path / "plan.json"
    write_recovery_plan(path, plan)
    ledger.reserve(kind="vision", model="gpt-6-luna", amount=9.7)
    client = Client("success")
    result = transcribe_inventory(manifest_path=manifest, ledger=ledger, client=client, output_dir=tmp_path / "new",
        batch_size=4, unknown_transport_new_attempt_plan=path, acknowledge_held_parent_reservations=True)
    assert result["states"] == {"failed_or_unknown": 1} and not client.calls
    with sqlite3.connect(ledger.path) as db:
        row = db.execute("SELECT status,charged,reserved FROM calls WHERE id=?",
                         (plan["parents"][0]["held_budget_call_id"],)).fetchone()
    assert row[0] == "unknown_cost" and row[1] == row[2] > 0


def test_safe_error_fields_never_access_body_headers_or_message():
    class Error(Exception):
        status_code, code, param, request_id = 400, "invalid_json_schema", "text.format.schema", "req_fixture"

        @property
        def body(self):
            pytest.fail("Error bodies must never be read")

        @property
        def headers(self):
            pytest.fail("Error headers must never be read")

    assert safe_error_fields(Error("private provider message")) == {"http_status": 400,
        "code": "invalid_json_schema", "param": "text.format.schema", "request_id": "req_fixture"}
    error = Error()
    error.code, error.param, error.request_id = "sk-proj-fixture", "unregistered.private_field", "unregistered"
    assert safe_error_fields(error) == {"http_status": 400}
