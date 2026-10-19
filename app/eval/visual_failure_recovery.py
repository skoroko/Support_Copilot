"""Offline, explicit plans for known paid visual failures; never retry transport errors."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from app.eval.acceptance_budget import usage_cost
from app.eval.visual_transcribe import (
    canonical_hash,
    crop_bytes,
    load_inventory,
    pending_records,
    reviewed_region_ids,
    sha256,
)


def _read_attempt(path: Path, attempt_id: str) -> tuple[dict, dict]:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        run = db.execute("SELECT identity FROM run WHERE id=1").fetchone()
        row = db.execute("SELECT state,result FROM attempts WHERE id=?", (attempt_id,)).fetchone()
    if not run or not row:
        raise ValueError("parent run and exact attempt required")
    identity, result = json.loads(run[0]), json.loads(row[1])
    if result.get("batch_id") != attempt_id or result.get("state") != row[0]:
        raise ValueError("parent attempt state or identity mismatch")
    if attempt_id != canonical_hash([identity, result.get("group_ids")]):
        raise ValueError("parent attempt fingerprint mismatch")
    return identity, result


def _attempt_groups(identity: dict, result: dict, groups: list[dict]) -> list[dict]:
    by_id = {group["group_id"]: group for group in groups}
    all_regions = {row["region_id"] for group in groups for row in group["occurrences"]}
    excluded = identity.get("excluded_region_ids", [])
    group_ids, region_ids = result.get("group_ids"), result.get("region_ids")
    if (not isinstance(excluded, list) or len(set(excluded)) != len(excluded)
            or not set(excluded) <= all_regions
            or not isinstance(group_ids, list) or not group_ids or len(set(group_ids)) != len(group_ids)
            or not set(group_ids) <= set(by_id) or not isinstance(region_ids, list)
            or len(set(region_ids)) != len(region_ids)):
        raise ValueError("exact unique parent inventory scope required")
    allowed = all_regions - set(excluded)
    if recovery := identity.get("failed_known_recovery") or identity.get("unknown_transport_new_attempt"):
        allowed &= {region for group in recovery["groups"] for region in group["region_ids"]}
    selected = [{**by_id[group_id], "occurrences": [row for row in by_id[group_id]["occurrences"]
        if row["region_id"] in allowed]} for group_id in group_ids]
    expected = {row["region_id"] for group in selected for row in group["occurrences"]}
    if set(region_ids) != expected or any(not group["occurrences"] for group in selected):
        raise ValueError("parent occurrence scope mismatch")
    return selected


def _known_reason(result: dict) -> str:
    response = result.get("response") or {}
    if (result.get("state") == "failed_or_unknown" and result.get("error_type") == "JSONDecodeError"
            and response.get("status") == "completed"):
        try:
            json.loads(response.get("output_text", ""))
        except json.JSONDecodeError:
            return "completed_invalid_json"
    if (result.get("state") == "incomplete_pending_review" and response.get("status") == "incomplete"
            and (response.get("incomplete_details") or {}).get("reason") == "max_output_tokens"):
        return "incomplete_output_token_limit"
    raise ValueError("only persisted completed JSON errors or known output-token truncation are recoverable")


def build_recovery_plan(*, manifest_path: Path, ledger_path: Path,
                        attempts: list[tuple[Path, str]], batch_size: int = 4) -> dict:
    """Pure read-only operation. A new plan authorizes new paid calls, not replay."""
    if batch_size not in {4, 8} or not attempts:
        raise ValueError("explicit parent attempts and recovery batches of four or eight required")
    inventory = load_inventory(manifest_path)
    manifest_hash = sha256(manifest_path.read_bytes())
    parents, selected, seen_parents, seen_regions, seen_groups = [], [], set(), set(), set()
    for journal_path, attempt_id in attempts:
        parent_key = (str(journal_path.resolve()), attempt_id)
        if parent_key in seen_parents:
            raise ValueError("duplicate parent attempt")
        seen_parents.add(parent_key)
        identity, result = _read_attempt(journal_path, attempt_id)
        if (identity.get("manifest_sha256") != manifest_hash or identity.get("mode") != "full"
                or identity.get("model") != "gpt-6-luna"
                or identity.get("budget_ledger") != str(ledger_path.resolve())
                or result.get("provider_request_started") is not True
                or result.get("pending_records")):
            raise ValueError("parent source, budget or completed-response scope mismatch")
        reason = _known_reason(result)
        response = result["response"]
        usage = response.get("usage")
        request_hash = result.get("request_fingerprint")
        if (not isinstance(response.get("id"), str) or not response["id"]
                or not isinstance(usage, dict) or not isinstance(request_hash, str) or len(request_hash) != 64):
            raise ValueError("persisted response identity, usage and request fingerprint required")
        usage_cost(identity["model"], usage)
        started = datetime.fromisoformat(result["started_at"]).timestamp()
        finished = datetime.fromisoformat(result["finished_at"]).timestamp()
        if finished < started:
            raise ValueError("invalid parent call interval")
        with closing(sqlite3.connect(ledger_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            candidates = db.execute("SELECT id,usage,charged FROM calls WHERE kind='vision' AND model=? "
                "AND status='completed' AND fingerprint=? AND started>=? AND started<=?",
                (identity["model"], request_hash, started, finished)).fetchall()
        settled = [row for row in candidates if canonical_hash(json.loads(row[1])) == canonical_hash(usage)]
        if len(settled) != 1:
            raise ValueError("one exact completed paid reservation required; unknown calls cannot be recovered")
        parent_groups = _attempt_groups(identity, result, inventory["groups"])
        for group in parent_groups:
            regions = [row["region_id"] for row in group["occurrences"]]
            if group["group_id"] in seen_groups or seen_regions & set(regions):
                raise ValueError("overlapping parent recovery scopes")
            seen_groups.add(group["group_id"])
            seen_regions.update(regions)
            selected.append({"group_id": group["group_id"], "region_ids": regions,
                "occurrences_sha256": canonical_hash(group["occurrences"]), "parent_attempt_id": attempt_id})
        parents.append({"journal_path": str(journal_path.resolve()), "attempt_id": attempt_id,
            "run_sha256": canonical_hash(identity), "attempt_sha256": canonical_hash(result),
            "response_sha256": canonical_hash(response), "response_id": response["id"],
            "request_fingerprint": request_hash, "paid_budget_call_id": settled[0][0],
            "paid_usage": usage, "paid_charged_micros": settled[0][2], "reason": reason})
    plan = {"schema_version": 1, "purpose": "explicit_known_visual_failure_new_paid_run",
        "manifest_sha256": manifest_hash, "budget_ledger": str(ledger_path.resolve()),
        "batch_size": batch_size, "parents": parents, "groups": selected,
        "approval_status": "pending", "semantic_verified": False}
    return {**plan, "plan_sha256": canonical_hash(plan)}


def write_recovery_plan(path: Path, plan: dict) -> None:
    """Exclusive creation prevents silently rewriting an approved immutable plan."""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")


def build_unknown_transport_plan(*, manifest_path: Path, ledger_path: Path, data_root: Path,
                                 attempts: list[tuple[Path, str]], batch_size: int = 4,
                                 acknowledge_held_parent_reservations: bool = False) -> dict:
    """Explicitly authorize NEW calls after terminal unknown transport outcomes.

    No old reservation is released, repriced or reused. A server rejection,
    received response or still-reserved/in-flight call is never eligible here.
    """
    from app.services.corpus import file_hash, source_identity

    if acknowledge_held_parent_reservations is not True or batch_size not in {4, 8} or not attempts:
        raise ValueError("Explicit acknowledgment of full held parent charges and bounded new calls required")
    inventory = load_inventory(manifest_path)
    manifest_hash = sha256(manifest_path.read_bytes())
    parents, selected, source_bindings = [], [], {}
    seen_parents, seen_groups, seen_regions = set(), set(), set()
    for journal_path, attempt_id in attempts:
        key = (str(journal_path.resolve()), attempt_id)
        if key in seen_parents:
            raise ValueError("Duplicate transport parent attempt")
        seen_parents.add(key)
        identity, result = _read_attempt(journal_path, attempt_id)
        if (identity.get("manifest_sha256") != manifest_hash or identity.get("mode") != "full"
                or identity.get("model") != "gpt-6-luna"
                or identity.get("budget_ledger") != str(ledger_path.resolve())
                or result.get("state") != "failed_or_unknown"
                or result.get("error_type") != "APIConnectionError"
                or result.get("provider_request_started") is not True
                or result.get("response") or result.get("pending_records")
                or (result.get("error_fields") or {}).get("http_status") is not None):
            raise ValueError("Only terminal APIConnectionError without any provider response is eligible")
        request_hash = result.get("request_fingerprint")
        if not isinstance(request_hash, str) or len(request_hash) != 64 or not result.get("finished_at"):
            raise ValueError("Exact terminal parent fingerprint and finish time required")
        started = datetime.fromisoformat(result["started_at"]).timestamp()
        finished = datetime.fromisoformat(result["finished_at"]).timestamp()
        if finished < started:
            raise ValueError("Invalid terminal parent call interval")
        with closing(sqlite3.connect(ledger_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT id,status,charged,reserved,usage FROM calls WHERE kind='vision' AND model=? "
                "AND fingerprint=? AND started>=? AND started<=?",
                (identity["model"], request_hash, started, finished)).fetchall()
        if (len(rows) != 1 or rows[0][1] != "unknown_cost" or rows[0][2] != rows[0][3]
                or type(rows[0][3]) is not int or rows[0][3] <= 0 or rows[0][4] is not None):
            raise ValueError("One terminal unknown-cost parent with its FULL reservation still held required")
        parent_groups = _attempt_groups(identity, result, inventory["groups"])
        for group in parent_groups:
            regions = [row["region_id"] for row in group["occurrences"]]
            if group["group_id"] in seen_groups or set(regions) & seen_regions:
                raise ValueError("Overlapping transport parent inventory scopes")
            seen_groups.add(group["group_id"])
            seen_regions.update(regions)
            crop_data, crop_path = crop_bytes(group, manifest_path, identity["crop"])
            for occurrence in group["occurrences"]:
                relative = occurrence["relative_path"]
                if ("\\" in relative or ":" in relative or relative.startswith("/")
                        or any(piece in {"", ".", ".."} for piece in relative.split("/"))):
                    raise ValueError("Transport recovery source must stay inside permitted data root")
                source = (data_root.resolve() / relative).resolve()
                binding = {key: occurrence[key] for key in ("relative_path", "source_id", "source_hash")}
                if (not source.is_relative_to(data_root.resolve()) or source.suffix.lower() != ".pdf"
                        or not source.is_file() or source_identity(source, data_root)[0] != binding["source_id"]
                        or file_hash(source) != binding["source_hash"]
                        or relative in source_bindings and source_bindings[relative] != binding):
                    raise ValueError("Transport recovery original source identity or bytes changed")
                source_bindings[relative] = binding
            selected.append({"group_id": group["group_id"], "region_ids": regions,
                "occurrences_sha256": canonical_hash(group["occurrences"]), "parent_attempt_id": attempt_id,
                "crop": identity["crop"], "crop_path": str(crop_path), "crop_sha256": sha256(crop_data)})
        parents.append({"journal_path": str(journal_path.resolve()), "attempt_id": attempt_id,
            "run_sha256": canonical_hash(identity), "attempt_sha256": canonical_hash(result),
            "request_fingerprint": request_hash, "held_budget_call_id": rows[0][0],
            "held_status": rows[0][1], "held_charged_micros": rows[0][2], "held_reserved_micros": rows[0][3],
            "reason": "terminal_unknown_transport_new_call_old_full_charge_retained"})
    plan = {"schema_version": 1, "purpose": "explicit_terminal_unknown_transport_new_paid_run",
        "manifest_sha256": manifest_hash, "budget_ledger": str(ledger_path.resolve()),
        "data_root": str(data_root.resolve()), "batch_size": batch_size, "parents": parents, "groups": selected,
        "source_bindings": [source_bindings[key] for key in sorted(source_bindings)],
        "held_parent_reservations_acknowledged": True,
        "old_reservations_released": False, "approval_status": "pending", "semantic_verified": False}
    return {**plan, "plan_sha256": canonical_hash(plan)}


def load_unknown_transport_plan(path: Path, *, manifest_path: Path, ledger_path: Path,
                                output_dir: Path, batch_size: int, crop: str,
                                acknowledge_held_parent_reservations: bool) -> tuple[list[dict], dict]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    attempts = [(Path(row["journal_path"]), row["attempt_id"]) for row in plan["parents"]]
    if any(journal.resolve() == (output_dir / "attempts.sqlite3").resolve() for journal, _ in attempts):
        raise ValueError("Transport new-attempt plan requires a separate new output journal")
    current = build_unknown_transport_plan(manifest_path=manifest_path, ledger_path=ledger_path,
        data_root=Path(plan["data_root"]), attempts=attempts, batch_size=batch_size,
        acknowledge_held_parent_reservations=acknowledge_held_parent_reservations)
    if canonical_hash(current) != canonical_hash(plan) or any(row["crop"] != crop for row in plan["groups"]):
        raise ValueError("Immutable transport plan, held charge, source or crop binding changed")
    by_id = {group["group_id"]: group for group in load_inventory(manifest_path)["groups"]}
    groups = [{**by_id[row["group_id"]], "occurrences": [occ for occ in by_id[row["group_id"]]["occurrences"]
        if occ["region_id"] in set(row["region_ids"])]} for row in plan["groups"]]
    return groups, plan


def load_recovery_plan(path: Path, *, manifest_path: Path, ledger_path: Path,
                       output_dir: Path, batch_size: int) -> tuple[list[dict], dict]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    attempts = [(Path(row["journal_path"]), row["attempt_id"]) for row in plan["parents"]]
    if any(journal.resolve() == (output_dir / "attempts.sqlite3").resolve() for journal, _ in attempts):
        raise ValueError("recovery requires a separate new output journal")
    current = build_recovery_plan(manifest_path=manifest_path, ledger_path=ledger_path,
                                  attempts=attempts, batch_size=batch_size)
    if canonical_hash(current) != canonical_hash(plan):
        raise ValueError("immutable recovery plan or parent/source/budget binding changed")
    by_id = {group["group_id"]: group for group in load_inventory(manifest_path)["groups"]}
    groups = [{**by_id[row["group_id"]], "occurrences": [occ for occ in by_id[row["group_id"]]["occurrences"]
        if occ["region_id"] in set(row["region_ids"])]} for row in plan["groups"]]
    return groups, plan


def coverage_report(*, manifest_path: Path, journals: list[Path],
                    approved_path: Path | None = None) -> dict:
    """Validated candidate availability is explicitly separate from reviewed recovery."""
    inventory = load_inventory(manifest_path)
    groups = inventory["groups"]
    manifest_hash = sha256(manifest_path.read_bytes())
    all_regions = {row["region_id"] for group in groups for row in group["occurrences"]}
    approved = reviewed_region_ids(approved_path, groups)
    buckets = {key: set() for key in ("pending_candidate", "failed_known", "unknown", "provider_rejected_unpriced", "in_flight")}
    attempts, duplicate_candidates, ledger_paths = [], set(), set()
    for path in journals:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("BEGIN")
            ids = [row[0] for row in db.execute("SELECT id FROM attempts ORDER BY rowid")]
        for attempt_id in ids:
            identity, result = _read_attempt(path, attempt_id)
            if identity.get("manifest_sha256") != manifest_hash or identity.get("mode") != "full":
                raise ValueError("coverage journal source manifest mismatch")
            ledger_paths.add(identity["budget_ledger"])
            selected = _attempt_groups(identity, result, groups)
            regions = set(result["region_ids"])
            if result["state"] == "pending_review":
                expected = pending_records(json.loads(result["response"]["output_text"]), selected,
                                           crop=identity["crop"], batch_id=attempt_id)
                actual = [{key: value for key, value in row.items() if key not in {"resolves_attempt", "failed_known_recovery", "unknown_transport_new_attempt"}}
                          for row in result.get("pending_records", [])]
                if canonical_hash(expected) != canonical_hash(actual):
                    raise ValueError("persisted candidate records mismatch exact response and inventory")
                duplicate_candidates |= buckets["pending_candidate"] & regions
                bucket = "pending_candidate"
            elif result["state"] in {"attempted_unknown", "response_received_pending_parse"}:
                bucket = "in_flight"
            elif (result.get("error_type") in {"BadRequestError", "AuthenticationError", "PermissionDeniedError",
                    "NotFoundError", "UnprocessableEntityError"}
                    or type((result.get("error_fields") or {}).get("http_status")) is int
                    and 400 <= result["error_fields"]["http_status"] < 500):
                bucket = "provider_rejected_unpriced"
            else:
                try:
                    _known_reason(result)
                    # A raw response alone does not establish paid completion.
                    build_recovery_plan(manifest_path=manifest_path, ledger_path=Path(identity["budget_ledger"]),
                                        attempts=[(path, attempt_id)])
                    bucket = "failed_known"
                except ValueError:
                    bucket = "unknown"
            buckets[bucket] |= regions
            attempts.append({"journal_path": str(path.resolve()), "attempt_id": attempt_id,
                "state": result["state"], "category": bucket, "groups": len(selected),
                "occurrences": len(regions), "error_type": result.get("error_type")})
    if len(ledger_paths) > 1:
        raise ValueError("coverage journals must share one task budget")
    available = approved | buckets["pending_candidate"]
    remaining = all_regions - available
    unresolved = {}
    for category in ("provider_rejected_unpriced", "unknown", "failed_known", "in_flight"):
        unresolved[category] = buckets[category] & remaining
        remaining -= unresolved[category]
    unresolved["unattempted"] = remaining
    summary = {"inventory_groups": len(groups), "inventory_occurrences": len(all_regions),
        "pending_candidate_occurrences": len(buckets["pending_candidate"]),
        "prior_reviewed_occurrences": len(approved), "candidate_or_prior_review_occurrences": len(available),
        "all_occurrences_have_candidate_or_prior_review": available == all_regions,
        "fully_accounted_groups": sum({row["region_id"] for row in group["occurrences"]} <= available for group in groups),
        "duplicate_candidate_occurrences": len(duplicate_candidates), "attempted_batches": len(attempts),
        "source_recovery_verified": False, "candidate_approval_status": "pending"}
    for category, regions in unresolved.items():
        summary[category + "_occurrences"] = len(regions)
        summary[category + "_groups"] = sum(bool(regions & {row["region_id"] for row in group["occurrences"]}) for group in groups)
    return {"schema_version": 1, "read_only": True, "manifest_sha256": manifest_hash,
        "summary": summary, "attempts": attempts,
        "unresolved_region_ids": {key: sorted(value) for key, value in unresolved.items()}}
