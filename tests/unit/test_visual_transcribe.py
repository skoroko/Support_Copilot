from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.eval.acceptance_budget import BudgetLedger
from app.eval.visual_transcribe import (
    AttemptJournal,
    build_request,
    crop_bytes,
    load_inventory,
    pending_records,
    response_schema,
    reviewed_region_ids,
    sha256,
    transcribe_inventory,
    triage_records,
    triage_schema,
    validate_official_endpoint,
)


def inventory(tmp_path, count=2, duplicate_occurrence=False):
    groups = []
    for number in range(count):
        image = b"\x89PNG\r\n\x1a\n" + bytes([number])
        path = tmp_path / f"crop{number}.png"
        path.write_bytes(image)
        occurrence = {"region_id": f"region{number}", "source_hash": f"source{number}",
            "source_id": f"sourceid{number}", "relative_path": f"manual{number}.pdf", "location_id": f"location{number}",
            "page": 1, "bbox": [0., 0., 10., 10.], "image_sha256": f"image{number}", "product": "gts_nx", "version": "unknown"}
        occurrences = [occurrence]
        if duplicate_occurrence:
            occurrences.append({**occurrence, "region_id": f"other{number}", "source_hash": "other_source",
                                "relative_path": "other_manual.pdf", "location_id": "other_location"})
        groups.append({"group_id": f"group{number}", "image_sha256": f"image{number}",
            "representative_region_id": f"region{number}", "occurrences": occurrences,
            "full_crop_path": str(path), "full_crop_sha256": sha256(image),
            "classifier_crop_path": str(path), "classifier_crop_sha256": sha256(image)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"complete": True, "groups": groups}), encoding="utf-8")
    return manifest, groups


def candidate_payload(groups):
    return {"groups": [{"group_id": group["group_id"], "classification": "formula", "occurrences": [
        {"region_id": row["region_id"], "kind": "formula", "literal_transcription": f"x − [{row['region_id']}]",
         "uncertainty": "partial", "uncertain_tokens": ["[?]"], "warnings": ["ambiguous subscript"]}
        for row in group["occurrences"]]} for group in groups]}


class FakeClient:
    base_url = "https://api.openai.com/v1/"
    max_retries = 0
    def __init__(self, groups, *, failure=False, incomplete=False, malformed=False):
        self.groups = {group["group_id"]: group for group in groups}
        self.failure, self.incomplete, self.malformed = failure, incomplete, malformed
        self.calls = []
        self.responses = SimpleNamespace(create=self.create)
    def create(self, **request):
        self.calls.append(request)
        if self.failure:
            raise TimeoutError("secret raw provider detail")
        content = request["input"][0]["content"]
        if request["text"]["format"]["name"] == "visual_presence_triage":
            tiles = json.loads(content[0]["text"])["tiles"]
            payload = {"groups": [{"group_id": row["group_id"], "classification": "formula",
                "important_visible_text": True, "uncertainty": "partial", "warnings": []} for row in tiles]}
        else:
            labels = [json.loads(row["text"]) for row in content if row["type"] == "input_text"]
            payload = candidate_payload([{**self.groups[row["group_id"]], "occurrences": row["occurrences"]}
                                         for row in labels])
        output = "not JSON" if self.malformed else json.dumps(payload)
        usage = {"input_tokens": 100, "output_tokens": 100, "total_tokens": 200}
        return SimpleNamespace(id="response_id", status="incomplete" if self.incomplete else "completed",
            output_text=output, output=[], incomplete_details=None,
            usage=SimpleNamespace(model_dump=lambda: usage))


@pytest.mark.parametrize("url", ["http://api.openai.com/v1", "https://evil.example/v1", "https://api.openai.com.evil/v1",
    "https://token@api.openai.com/v1", "https://api.openai.com/v1?token=value"])
def test_official_endpoint_only(url):
    with pytest.raises(ValueError):
        validate_official_endpoint(url)


def test_inventory_requires_complete_and_unique_occurrence_bindings(tmp_path):
    manifest, _groups = inventory(tmp_path)
    assert len(load_inventory(manifest)["groups"]) == 2
    payload = json.loads(manifest.read_text())
    payload["complete"] = False
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="complete"):
        load_inventory(manifest)
    payload["complete"] = True
    payload["groups"][1]["occurrences"][0]["region_id"] = "region0"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_inventory(manifest)


def test_crop_hash_and_native_context_do_not_substitute_for_visible_pixels(tmp_path):
    manifest, groups = inventory(tmp_path, 1, duplicate_occurrence=True)
    request = build_request(groups, manifest_path=manifest, contexts={"location0": "native context outside crop"},
        model="gpt-6-luna", crop="full", detail="high", effort="none", output_tokens=8192)
    inputs = request["input"][0]["content"]
    assert len([row for row in inputs if row["type"] == "input_image"]) == 1
    context = json.loads(inputs[0]["text"])["occurrences"]
    assert len(context) == 2 and context[1]["representative_crop_only"] is True
    assert "NOT image OCR" in request["instructions"]
    Path(groups[0]["full_crop_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        crop_bytes(groups[0], manifest, "full")


def test_each_duplicate_occurrence_is_distinct_pending_review_never_autoapproved(tmp_path):
    _, groups = inventory(tmp_path, 1, duplicate_occurrence=True)
    records = pending_records(candidate_payload(groups), groups, crop="full", batch_id="batch")
    assert len(records) == 2 and records[0]["text"] != records[1]["text"]
    assert records[0]["source_hash"] != records[1]["source_hash"]
    assert all(row["approval_status"] == "pending" and row["reviewer"] == "" and row["verified"] is False for row in records)
    assert all(row["candidate_from_representative_and_occurrence_unverified"] for row in records)
    payload = candidate_payload(groups)
    payload["groups"][0]["occurrences"].pop()
    with pytest.raises(ValueError, match="omitted occurrences"):
        pending_records(payload, groups, crop="full", batch_id="batch")


def test_resume_only_unattempted_and_actual_usage_shared_budget(tmp_path):
    manifest, groups = inventory(tmp_path)
    ledger = BudgetLedger(tmp_path / "budget.sqlite3")
    client = FakeClient(groups)
    output = tmp_path / "full"
    first = transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger, client=client, batch_size=1, max_batches=1)
    second = transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger, client=client, batch_size=1)
    third = transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger, client=client, batch_size=1)
    assert first["attempted_now"] == second["attempted_now"] == 1
    assert third["attempted_now"] == 0 and len(client.calls) == 2
    assert ledger.summary()["calls"] == 2 and ledger.summary()["by_kind"]["vision"] > 0
    records = json.loads((output / "pending_transcriptions.json").read_text())["records"]
    assert len(records) == 2
    with pytest.raises(ValueError, match="configuration mismatch"):
        transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger, client=client, batch_size=2)


@pytest.mark.parametrize("failure,incomplete,malformed", [(True, False, False), (False, True, False), (False, False, True)])
def test_failed_unknown_incomplete_and_malformed_calls_persist_never_retry(tmp_path, failure, incomplete, malformed):
    manifest, groups = inventory(tmp_path, 1)
    ledger = BudgetLedger(tmp_path / "budget.sqlite3")
    client = FakeClient(groups, failure=failure, incomplete=incomplete, malformed=malformed)
    output = tmp_path / "run"
    transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger, client=client)
    transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger, client=client)
    assert len(client.calls) == 1
    journal = AttemptJournal(output / "attempts.sqlite3", json.loads(__import__("sqlite3").connect(output / "attempts.sqlite3")
        .execute("SELECT identity FROM run").fetchone()[0]))
    result = journal.results()[0]
    assert result["pending_records"] == []
    assert "secret raw provider detail" not in json.dumps(result)
    if failure:
        assert ledger.summary()["unknown_cost_calls"] == 1
    else:
        assert result["response"]["usage"]["total_tokens"] == 200
        assert result["response"]["status"] in {"completed", "incomplete"}


def test_contact_triage_covers_all_groups_classification_only(tmp_path):
    manifest, groups = inventory(tmp_path)
    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(b"\x89PNG\r\n\x1a\ncontact")
    contacts = tmp_path / "contacts.json"
    contacts.write_text(json.dumps({"sheets": [{"path": str(sheet), "sha256": sha256(sheet.read_bytes()),
        "tiles": [{"tile_id": f"V{number}", "group_id": group["group_id"],
            "representative_region_id": group["representative_region_id"], "occurrences": group["occurrences"],
            "sheet_bbox": [0, 0, 10, 10]} for number, group in enumerate(groups)]}]}))
    ledger, client = BudgetLedger(tmp_path / "budget.sqlite3"), FakeClient(groups)
    report = transcribe_inventory(manifest_path=manifest, output_dir=tmp_path / "triage", ledger=ledger,
        client=client, mode="triage", contact_path=contacts, output_tokens=4096)
    assert report["classified_unique_images"] == 2 and report["pending_occurrences"] == 0
    triage = json.loads((tmp_path / "triage" / "triage_classifications.json").read_text())
    assert all(row["classification_only"] and not row["literal_transcription_performed"] for row in triage["records"])
    assert len(client.calls) == 1 and len(client.calls[0]["input"][0]["content"]) == 2
    schema = client.calls[0]["text"]["format"]["schema"]["properties"]["groups"]
    assert schema["minItems"] == schema["maxItems"] == 2
    assert schema["items"]["properties"]["group_id"]["enum"] == ["group0", "group1"]


def test_raw_response_is_durable_even_when_usage_settlement_fails(tmp_path, monkeypatch):
    manifest, groups = inventory(tmp_path, 1)
    ledger, client = BudgetLedger(tmp_path / "budget.sqlite3"), FakeClient(groups)
    def fail_settlement(*args, **kwargs):
        raise ValueError("private settlement detail")
    monkeypatch.setattr(ledger, "settle", fail_settlement)
    output = tmp_path / "raw"
    report = transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger, client=client)
    import sqlite3
    with sqlite3.connect(output / "attempts.sqlite3") as db:
        attempt = json.loads(db.execute("SELECT result FROM attempts").fetchone()[0])
    assert report["states"]["failed_or_unknown"] == 1
    assert attempt["response"]["output_text"]
    assert attempt["response"]["usage"]["total_tokens"] == 200
    assert "private settlement detail" not in json.dumps(attempt)
    assert ledger.summary()["unknown_cost_calls"] == 1


def test_full_selection_preserves_uncertain_and_missing_triage_and_rejects_wrong_source(tmp_path):
    manifest, groups = inventory(tmp_path, 3)
    triage_path = tmp_path / "triage.json"
    records = [{"group_id": group["group_id"], "classification": "decoration" if number == 0 else "uncertain",
                "important_visible_text": False, "uncertainty": "none" if number == 0 else "partial",
                "image_sha256": group["image_sha256"], "region_ids": [row["region_id"] for row in group["occurrences"]]}
               for number, group in enumerate(groups[:2])]
    triage_path.write_text(json.dumps({"classification_only": True, "records": records}))
    ledger, client = BudgetLedger(tmp_path / "budget.sqlite3"), FakeClient(groups)
    report = transcribe_inventory(manifest_path=manifest, output_dir=tmp_path / "selected", ledger=ledger,
        client=client, triage_results_path=triage_path)
    assert report["selected_unique_images"] == 2
    assert report["pending_occurrences"] == 2
    records[0]["image_sha256"] = "wrong_scope"
    triage_path.write_text(json.dumps({"classification_only": True, "records": records}))
    with pytest.raises(ValueError, match="binding mismatch"):
        transcribe_inventory(manifest_path=manifest, output_dir=tmp_path / "wrong", ledger=ledger,
            client=client, triage_results_path=triage_path)


def test_full_schema_binds_exact_batch_and_each_groups_own_occurrences(tmp_path):
    _, groups = inventory(tmp_path, 2, duplicate_occurrence=True)
    groups[1]["occurrences"].pop()
    schema = response_schema(groups)["properties"]["groups"]
    assert schema["minItems"] == schema["maxItems"] == 2
    variants = schema["items"]["anyOf"]
    assert len(variants) == 2
    for group, variant in zip(groups, variants, strict=True):
        properties = variant["properties"]
        assert properties["group_id"]["enum"] == [group["group_id"]]
        occurrences = properties["occurrences"]
        assert occurrences["minItems"] == occurrences["maxItems"] == len(group["occurrences"])
        assert occurrences["items"]["properties"]["region_id"]["enum"] == [r["region_id"] for r in group["occurrences"]]
    payload = candidate_payload(groups)
    payload["groups"][1]["occurrences"][0]["region_id"] = "region0"
    with pytest.raises(ValueError, match="identity mismatch"):
        pending_records(payload, groups, crop="full", batch_id="misbound")
    duplicate = candidate_payload(groups)
    duplicate["groups"][1] = copy.deepcopy(duplicate["groups"][0])
    with pytest.raises(ValueError, match="identity mismatch"):
        pending_records(duplicate, groups, crop="full", batch_id="duplicate")


def test_triage_schema_uses_exact_sheet_size_including_short_final_sheet(tmp_path):
    _, groups = inventory(tmp_path, 20)
    for batch in (groups, groups[:11]):
        array = triage_schema(batch)["properties"]["groups"]
        assert array["minItems"] == array["maxItems"] == len(batch)
        assert array["items"]["properties"]["group_id"]["enum"] == [g["group_id"] for g in batch]
    payload = {"groups": [{"group_id": "group0", "classification": "formula",
        "important_visible_text": True, "uncertainty": "none", "warnings": []}] * 20}
    with pytest.raises(ValueError, match="misbound"):
        triage_records(payload, groups, batch_id="duplicate")


def approved_record(occurrence):
    return {**occurrence, "approval_status": "approved", "kind": "formula", "text": "x = 1",
            "reviewer": "independent-reviewer", "reviewed_at": "2026-10-16", "review_id": "exact-review"}


def test_review_exclusion_is_exact_occurrence_only_not_whole_duplicate_group(tmp_path):
    manifest, groups = inventory(tmp_path, 2, duplicate_occurrence=True)
    reviews = tmp_path / "reviews.json"
    rows = [approved_record(groups[0]["occurrences"][0]),
            {**approved_record(groups[1]["occurrences"][0]), "approval_status": "pending"}]
    reviews.write_text(json.dumps(rows))
    ledger, client = BudgetLedger(tmp_path / "budget.sqlite3"), FakeClient(groups)
    report = transcribe_inventory(manifest_path=manifest, output_dir=tmp_path / "excluded", ledger=ledger,
        client=client, exclude_reviewed_transcriptions=reviews)
    assert report["excluded_reviewed_occurrences"] == 1
    assert report["selected_unique_images"] == 2 and report["selected_occurrences"] == 3
    inputs = client.calls[0]["input"][0]["content"]
    first = json.loads(inputs[0]["text"])
    assert first["representative_region_id"] == "region0"
    assert [r["region_id"] for r in first["occurrences"]] == ["other0"]
    records = json.loads((tmp_path / "excluded/pending_transcriptions.json").read_text())["records"]
    assert {r["region_id"] for r in records} == {"other0", "region1", "other1"}
    assert all(r["approval_status"] == "pending" and not r["verified"] for r in records)
    rows[0]["source_hash"] = "other-source"
    reviews.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="binding mismatch"):
        reviewed_region_ids(reviews, groups)


def test_all_reviewed_occurrences_skip_image_without_call_or_approval_propagation(tmp_path):
    manifest, groups = inventory(tmp_path, 1)
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps({"records": [approved_record(groups[0]["occurrences"][0])]}))
    ledger, client = BudgetLedger(tmp_path / "budget.sqlite3"), FakeClient(groups)
    report = transcribe_inventory(manifest_path=manifest, output_dir=tmp_path / "excluded", ledger=ledger,
        client=client, exclude_reviewed_transcriptions=reviews)
    assert report["attempted_now"] == report["selected_unique_images"] == report["pending_occurrences"] == 0
    assert not client.calls and ledger.summary()["calls"] == 0
    rows = json.loads(reviews.read_text())["records"]
    rows[0]["review_id"] = ""
    reviews.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="independent"):
        reviewed_region_ids(reviews, groups)


@pytest.mark.parametrize("mutated_id", [False, True])
def test_explicit_new_run_resolves_only_missing_groups_and_preserves_prior_attempt(tmp_path, mutated_id):
    import sqlite3

    manifest, groups = inventory(tmp_path, 3)
    ledger = BudgetLedger(tmp_path / "budget.sqlite3")
    unrelated = ledger.reserve(kind="generation", model="gpt-6-luna", amount=.01)
    ledger.settle(unrelated, error=True)
    class MissingGroupClient(FakeClient):
        def create(self, **request):
            response = super().create(**request)
            payload = json.loads(response.output_text)
            payload["groups"] = payload["groups"][:1]
            if mutated_id:
                payload["groups"][0]["group_id"] = "groupX"
            response.output_text = json.dumps(payload)
            return response
    old = tmp_path / "old"
    report = transcribe_inventory(manifest_path=manifest, output_dir=old, ledger=ledger,
        client=MissingGroupClient(groups), batch_size=2)
    assert report["states"] == {"failed_or_unknown": 1}
    prior_path = old / "attempts.sqlite3"
    with sqlite3.connect(prior_path) as db:
        prior_id = db.execute("SELECT id FROM attempts").fetchone()[0]
    prior_hash = sha256(prior_path.read_bytes())
    client = FakeClient(groups)
    recovered = tmp_path / "resolved"
    report = transcribe_inventory(manifest_path=manifest, output_dir=recovered, ledger=ledger,
        client=client, resolves_attempt_journal=prior_path, resolves_attempt_id=prior_id)
    assert report["selected_unique_images"] == report["pending_occurrences"] == (2 if mutated_id else 1)
    records = json.loads((recovered / "pending_transcriptions.json").read_text())["records"]
    assert {r["region_id"] for r in records} == ({"region0", "region1"} if mutated_id else {"region1"})
    assert all(r["approval_status"] == "pending" for r in records)
    assert records[0]["resolves_attempt"]["prior_attempt_id"] == prior_id
    assert sha256(prior_path.read_bytes()) == prior_hash
    assert ledger.summary()["calls"] == 3 and ledger.summary()["unknown_cost_calls"] == 1
    transcribe_inventory(manifest_path=manifest, output_dir=recovered, ledger=ledger,
        client=client, resolves_attempt_journal=prior_path, resolves_attempt_id=prior_id)
    assert len(client.calls) == 1


@pytest.mark.parametrize("failure,incomplete,malformed", [(True, False, False), (False, True, False), (False, False, True)])
def test_explicit_resolution_cannot_replay_unknown_incomplete_or_unparseable_outputs(tmp_path, failure, incomplete, malformed):
    import sqlite3

    manifest, groups = inventory(tmp_path, 1)
    ledger = BudgetLedger(tmp_path / "budget.sqlite3")
    old = tmp_path / "old"
    transcribe_inventory(manifest_path=manifest, output_dir=old, ledger=ledger,
        client=FakeClient(groups, failure=failure, incomplete=incomplete, malformed=malformed))
    prior_path = old / "attempts.sqlite3"
    with sqlite3.connect(prior_path) as db:
        prior_id = db.execute("SELECT id FROM attempts").fetchone()[0]
    client = FakeClient(groups)
    with pytest.raises(ValueError):
        transcribe_inventory(manifest_path=manifest, output_dir=tmp_path / "invalid-resolution", ledger=ledger,
            client=client, resolves_attempt_journal=prior_path, resolves_attempt_id=prior_id)
    assert not client.calls


def test_prior_triage_skips_only_complete_sheets_preserving_bound_classifications(tmp_path):
    manifest, groups = inventory(tmp_path, 4)
    image = tmp_path / "sheet.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\ncontact")
    contacts = tmp_path / "contacts.json"
    sheets = [{"path": str(image), "sha256": sha256(image.read_bytes()), "tiles": [
        {"tile_id": f"V{group['group_id']}", "group_id": group["group_id"],
         "representative_region_id": group["representative_region_id"], "occurrences": group["occurrences"],
         "sheet_bbox": [0, 0, 10, 10]} for group in batch]} for batch in (groups[:2], groups[2:])]
    contacts.write_text(json.dumps({"sheets": sheets}))
    old_rows = triage_records({"groups": [{"group_id": group["group_id"], "classification": "formula",
        "important_visible_text": True, "uncertainty": "partial", "warnings": []} for group in groups[:3]]},
        groups[:3], batch_id="old-batch")
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"classification_only": True, "records": old_rows}))
    original_bytes = prior.read_bytes()
    ledger, client = BudgetLedger(tmp_path / "budget.sqlite3"), FakeClient(groups)
    output = tmp_path / "new-triage"
    report = transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger, client=client,
        mode="triage", contact_path=contacts, prior_triage_results=prior, output_tokens=4096)
    assert report["imported_classified_unique_images"] == 3
    assert report["classified_unique_images"] == 4 and report["attempted_now"] == 1
    assert len(client.calls) == 1
    labels = json.loads(client.calls[0]["input"][0]["content"][0]["text"])["tiles"]
    assert {row["group_id"] for row in labels} == {"group2", "group3"}
    records = json.loads((output / "triage_classifications.json").read_text())["records"]
    assert len(records) == 4
    by_id = {row["group_id"]: row for row in records}
    assert by_id["group0"]["imported_classification"]["prior_batch_id"] == "old-batch"
    assert by_id["group2"]["prior_classification"]["batch_id"] == "old-batch"
    assert prior.read_bytes() == original_bytes
    transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger, client=client,
        mode="triage", contact_path=contacts, prior_triage_results=prior, output_tokens=4096)
    assert len(client.calls) == 1
    old_rows[0]["occurrences"][0]["version"] = "wrong-version"
    prior.write_text(json.dumps({"classification_only": True, "records": old_rows}))
    with pytest.raises(ValueError, match="provenance"):
        transcribe_inventory(manifest_path=manifest, output_dir=tmp_path / "invalid", ledger=ledger, client=client,
            mode="triage", contact_path=contacts, prior_triage_results=prior, output_tokens=4096)


def test_three_shards_partition_fixed_global_batches_after_exact_exclusions(tmp_path):
    manifest, groups = inventory(tmp_path, 9, duplicate_occurrence=True)
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps([approved_record(row) for row in groups[0]["occurrences"]]
                                 + [approved_record(groups[2]["occurrences"][0])]))
    triage = tmp_path / "triage.json"
    triage.write_text(json.dumps({"classification_only": True, "records": [{
        "group_id": "group1", "classification": "decoration", "important_visible_text": False,
        "uncertainty": "none", "image_sha256": "image1", "region_ids": ["region1", "other1"]}]}))
    ledger = BudgetLedger(tmp_path / "budget.sqlite3")
    shard_regions, reports = [], []
    manifest_hash = sha256(manifest.read_bytes())
    for shard in range(3):
        output = tmp_path / f"shard{shard}"
        report = transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger,
            client=FakeClient(groups), batch_size=2, shard_count=3, shard_index=shard,
            exclude_reviewed_transcriptions=reviews, triage_results_path=triage)
        reports.append(report)
        rows = json.loads((output / "pending_transcriptions.json").read_text())["records"]
        shard_regions.append({row["region_id"] for row in rows})
        assert report["inventory_unique_images"] == 9 and report["inventory_occurrences"] == 18
        assert report["global_selected_unique_images"] == 7 and report["global_selected_occurrences"] == 13
        assert report["global_selected_batches"] == 4 and report["shard_count"] == 3 and report["shard_index"] == shard
        assert report["shard_selected_occurrences"] == len(rows)
    assert [report["shard_selected_batches"] for report in reports] == [2, 1, 1]
    assert [report["shard_selected_unique_images"] for report in reports] == [3, 2, 2]
    assert all(not shard_regions[a] & shard_regions[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    assert set.union(*shard_regions) == {row["region_id"] for group in groups[2:] for row in group["occurrences"]} - {"region2"}
    assert ledger.summary()["calls"] == 4
    assert sha256(manifest.read_bytes()) == manifest_hash


def test_shard_resume_is_bound_to_identity_and_invalid_shards_never_call(tmp_path):
    manifest, groups = inventory(tmp_path, 8)
    ledger, client = BudgetLedger(tmp_path / "budget.sqlite3"), FakeClient(groups)
    output = tmp_path / "shard0"
    first = transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger,
        client=client, batch_size=2, shard_count=3, shard_index=0, max_batches=1)
    second = transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger,
        client=client, batch_size=2, shard_count=3, shard_index=0)
    third = transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger,
        client=client, batch_size=2, shard_count=3, shard_index=0)
    assert first["attempted_now"] == second["attempted_now"] == 1 and third["attempted_now"] == 0
    assert len(client.calls) == 2
    with pytest.raises(ValueError, match="configuration mismatch"):
        transcribe_inventory(manifest_path=manifest, output_dir=output, ledger=ledger,
            client=client, batch_size=2, shard_count=3, shard_index=1)
    for count, index, mode in ((0, 0, "full"), (5, 0, "full"), (3, -1, "full"), (3, 3, "full"), (3, 0, "triage")):
        with pytest.raises(ValueError):
            transcribe_inventory(manifest_path=manifest, output_dir=tmp_path / "invalid", ledger=ledger,
                client=client, shard_count=count, shard_index=index, mode=mode, contact_path=manifest)
    assert len(client.calls) == 2
