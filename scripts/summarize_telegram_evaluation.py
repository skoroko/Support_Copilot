"""Aggregate saved Telegram runs after manual semantic review; no provider calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

NO_ANSWER = "В предоставленной базе знаний нет достаточно надёжных сведений для ответа на этот вопрос."
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {"summary.json", "comparison.json", "answers.csv"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = [value for value in values if value is not None]
    return {"denominator": len(values), "sum": sum(values),
            "mean": statistics.mean(values) if values else None,
            **{name: values.count(score) for name, score in (("full", 1), ("partial", .5), ("zero", 0))}}


def timing(rows, key="wall_latency_ms"):
    values = sorted(row[key] / 1000 for row in rows if isinstance(row.get(key), (int, float)))
    return {"n": len(values), "mean_s": statistics.mean(values) if values else None,
            "median_s": statistics.median(values) if values else None,
            "p95_nearest_rank_s": values[math.ceil(.95 * len(values)) - 1] if values else None,
            "max_s": max(values) if values else None}


def checks(rows, keys):
    return {key: {"observed": sum(type(row.get(key)) is bool for row in rows),
                  "passed": sum(row.get(key) is True for row in rows)} for key in keys}


def load(directory):
    run, review = read(directory / "run.json"), read(directory / "semantic_review.json")
    if review.get("run_id") != run.get("run_id"):
        raise ValueError("semantic review run_id mismatch")
    definitions = {row["id"]: row for row in run["scenarios"]}
    rows = {row["id"]: row for row in run["results"]}
    if (len(definitions) != len(run["scenarios"]) or len(rows) != len(run["results"])
            or rows.keys() != definitions.keys()):
        raise ValueError("scenario/results IDs must be unique and complete")
    entries = {}
    for identity, row in rows.items():
        if row["status"] not in {"completed", "failed"}:
            raise ValueError("unfinished ordinary result")
        entries[identity, "brief"] = {**row, "group": definitions[identity]["group"]}
        expansion = row.get("expansion") or {}
        if expansion.get("status") in {"started", "confirmation_started"}:
            raise ValueError("unfinished expansion")
        if expansion.get("status") == "completed":
            entries[identity, "expanded"] = {**expansion, **expansion["result"], "group": "expanded",
                                             "response": expansion["result"]["answer"]}
    scored = {}
    for item in review["results"]:
        key = item["id"], item["mode"]
        if not isinstance(item.get("notes"), str):
            raise TypeError("semantic review notes must be text")
        if key in scored or key not in entries:
            raise ValueError("duplicate or extra semantic review ID/mode")
        if not {"correctness", "grounding"} <= item.keys():
            raise ValueError("semantic review requires both score fields")
        for field in ("correctness", "grounding"):
            value = item.get(field)
            if (value is None and field == "grounding"):
                continue
            if type(value) not in {int, float} or value not in {0, .5, 1}:
                raise ValueError("scores must be 0, 0.5 or 1 (grounding may be null)")
        row = entries[key]
        refusal = row["status"] == "completed" and row.get("response") == NO_ANSWER and not row.get("sources")
        if row["status"] != "completed" and (item["correctness"] != 0 or item["grounding"] is not None):
            raise ValueError("failed result requires correctness 0 and grounding null")
        if refusal and item["grounding"] is not None:
            raise ValueError("genuine no-answer refusal requires grounding null")
        source_free_refusal = row["group"] == "refusal" and not row.get("sources")
        if (row["status"] == "completed" and not refusal and row["group"] != "clarification"
                and not source_free_refusal and item["grounding"] is None):
            raise ValueError("technical answer requires a grounding score")
        scored[key] = {**row, **item, "is_refusal": refusal}
    if scored.keys() != entries.keys():
        raise ValueError("missing semantic review ID/mode")
    checkpoint_path = directory / "checkpoints.json"
    checkpoint = read(checkpoint_path) if checkpoint_path.exists() else None
    if checkpoint and (checkpoint.get("run_id") != run["run_id"] or checkpoint.get("run_sha256") != sha(directory / "run.json")):
        raise ValueError("checkpoint run identity/hash mismatch")
    manifest_path = directory / "runtime_manifest.json"
    manifest = read(manifest_path) if manifest_path.exists() else run.get("runtime_manifest") or {}
    old_summary = read(directory / "summary.json") if (directory / "summary.json").exists() else {}
    anchor_path = directory / "baseline_manifest.json"
    anchor = read(anchor_path) if anchor_path.exists() else {}
    fingerprints = (run.get("dataset_sha256") or manifest.get("dataset_sha256")
                    or old_summary.get("dataset_sha256") or anchor.get("dataset_sha256"))
    for name, digest in (fingerprints or {}).items():
        path = ROOT / "tests" / "eval" / name
        if not path.exists() or sha(path) != digest:
            raise ValueError("dataset fingerprint mismatch: " + name)
    return run, list(scored.values()), checkpoint, manifest, fingerprints


def checkpoint_counts(checkpoint):
    if checkpoint is None:
        return None, {}
    names, followups, latest_calls = set(), {}, {}
    threads = checkpoint["threads"].values()
    for thread in threads:
        messages = thread.get("messages", [])
        groups, calls = [], []
        for message in messages:
            if message["type"] == "human":
                groups.append(calls)
                calls = []
            for call in message.get("tool_calls") or []:
                names.add(call["name"])
                calls.append(call["name"])
        groups.append(calls)
        latest_calls[thread.get("owner", "unknown")] = calls if any(message["type"] == "human" for message in messages) else None
        # Expanded turns have no tools; followup conversation groups align to
        # the collected scenario sequence before any optional expansion group.
        for index, identity in enumerate(thread.get("scenario_ids", []), start=1):
            followups[identity] = groups[index].count("search_knowledge_base") if index < len(groups) and thread.get("status") == "exported" else None
    return {"threads": len(checkpoint["threads"]),
            "missing": sum(thread.get("status") != "exported" for thread in checkpoint["threads"].values()),
            "observed_tool_names": sorted(names), "tool_calls_after_latest_human": latest_calls}, followups


def summarize(directory, loaded):
    run, scored, checkpoint, manifest, fingerprints = loaded
    brief = [row for row in scored if row["mode"] == "brief"]
    expanded = [row for row in scored if row["mode"] == "expanded"]
    groups = {}
    for row in brief:
        group = groups.setdefault(row["group"], {"planned": 0, "completed": 0, "failed": 0, "other": 0})
        group["planned"] += 1
        group[row["status"]] += 1
    cp_info, searches = checkpoint_counts(checkpoint)
    refused = [row for row in brief if row["group"] == "refusal" or
               (next(case for case in run["scenarios"] if case["id"] == row["id"]).get("retrieval_contract") or {}).get("expected_no_answer")]
    clarification = [row for row in brief if row["group"] == "clarification"]
    followup = [row for row in brief if row["group"] == "followup"]
    golden = [row for row in brief if row["group"] == "golden"]
    summary = {"run_id": run["run_id"], "version": {key: manifest.get(key) for key in
                ("code_root_git_head", "app_image", "bot_image")}, "collection": groups,
               "dataset_sha256": fingerprints, "checkpoints": cp_info,
               "golden_structural_proxies": run.get("structural_proxies", {}).get("golden_only", {}).get("metrics"),
               "golden_cited_answers": {"denominator": sum(row["status"] == "completed" and not row["is_refusal"] for row in golden),
                                         "with_sources": sum(bool(row.get("sources")) and not row["is_refusal"] for row in golden)},
               "refusals": {"denominator": len(refused), "correct": sum(row["correctness"] == 1 for row in refused),
                            "canonical_no_answer_without_sources": sum(row["is_refusal"] for row in refused),
                            "correctness_method": "semantic rubric for expected refusal scenarios",
                            "ids": [row["id"] for row in refused]},
               "clarification": {"denominator": len(clarification), "correct": sum(row["correctness"] == 1 for row in clarification),
                                 **checks(clarification, ["asks_missing_conditions"])},
               "fresh_search_after_followup": {"denominator": len(followup), "observed": sum(searches.get(row["id"]) is not None for row in followup),
                                               "confirmed": sum((searches.get(row["id"]) or 0) > 0 for row in followup),
                                               "counts": {row["id"]: searches.get(row["id"]) for row in followup}},
               "source_transport": checks(brief, ["transport_matches_persisted", "sources_match_persisted"]),
               "expansion_contract": {"denominator": len(expanded), **checks(expanded, ["preview_preserves_history", "false_confirmation_preserves_history", "replay_equal", "replay_preserves_history"]),
                                      "false_confirmation_422": {"observed": sum("false_confirmation_http_status" in row for row in expanded), "passed": sum(row.get("false_confirmation_http_status") == 422 for row in expanded)},
                                      "persisted_answer_matches": {"observed": sum(bool(row.get("persisted")) for row in expanded), "passed": sum(bool(row.get("persisted")) and row.get("answer") == row["persisted"].get("content") for row in expanded)},
                                      "persisted_sources_match": {"observed": sum(bool(row.get("persisted")) for row in expanded), "passed": sum(bool(row.get("persisted")) and row.get("sources") == row["persisted"].get("rag_sources") for row in expanded)},
                                      "replay_message_matches": {"observed": sum("replay_message_id" in row and "message_id" in row for row in expanded), "passed": sum("message_id" in row and row.get("replay_message_id") == row["message_id"] for row in expanded)}},
               "failed_requests": [{key: row.get(key) for key in ("id", "error", "wall_latency_ms")} for row in brief if row["status"] != "completed"]}
    for name, selected in (("golden", golden), ("followup", followup), ("expanded", expanded)):
        summary[name + "_correctness_and_completeness"] = stats([row["correctness"] for row in selected])
        summary[name + "_cited_fragment_grounding"] = stats([row["grounding"] for row in selected])
    expansions = [row["expansion"] for row in brief if row.get("expansion")]
    summary["expansion_collection"] = {"observed": len(expansions),
                                       "completed": sum(row["status"] == "completed" for row in expansions),
                                       "failed": sum(row["status"] == "failed" for row in expansions),
                                       "unavailable": sum(row["status"] == "unavailable" for row in expansions)}
    summary["timing"] = {"golden_all_including_failure": timing(golden), "golden_successful": timing([row for row in golden if row["status"] == "completed"]),
                         "golden_first_content_successful": timing([row for row in golden if row["status"] == "completed"], "ttft_ms"),
                         "followup": timing(followup), "expanded_confirmed_only": timing(expanded, "confirmed_latency_ms"),
                         "expanded_all_including_failure": timing(expansions),
                         "expanded_successful": timing([row for row in expansions if row["status"] == "completed"]),
                         "all_including_failure": timing(brief), "all_successful": timing([row for row in brief if row["status"] == "completed"])}
    pdf_path = directory / "pdf_pages.json"
    summary["pdf_page_audit"] = read(pdf_path) if pdf_path.exists() else None
    summary["expanded_detail_gain"] = {"denominator": len(expanded), "observed": sum("detail_gain" in row for row in expanded),
                                       "substantive": sum(row.get("detail_gain") == "substantive_details" for row in expanded),
                                       "mostly_repetition": sum(row.get("detail_gain") == "mostly_repetition" for row in expanded)}
    summary["artifact_sha256"] = {path.name: sha(path) for path in sorted(directory.iterdir())
                                  if path.is_file() and path.name not in OUTPUTS and path.suffix in {".json", ".csv"}}
    return summary


def compare(current, baseline):
    run, rows, _, manifest, datasets = current
    old_run, old_rows, _, old_manifest, old_datasets = baseline
    if run["scenarios"] != old_run["scenarios"]:
        raise ValueError("baseline scenarios differ")
    options = lambda value: {key: item for key, item in value.get("options", {}).items() if key != "api_url"}
    if options(run) != options(old_run):
        raise ValueError("baseline options differ")
    if datasets is None or old_datasets is None or datasets != old_datasets:
        raise ValueError("baseline dataset fingerprints absent or differ")
    for key in ("docstore", "config"):
        current_value, baseline_value = manifest.get(key), old_manifest.get(key)
        if key == "config":
            def normalized(config):
                values = dict(config or {})
                if "rag_min_score" in values:
                    if "rag_score_threshold" in values and values["rag_score_threshold"] != values["rag_min_score"]:
                        raise ValueError("conflicting threshold aliases")
                    values["rag_score_threshold"] = values.pop("rag_min_score")
                return values
            current_value, baseline_value = normalized(current_value), normalized(baseline_value)
        if not current_value or current_value != baseline_value:
            raise ValueError("baseline " + key + " differs or absent")
    old = {(row["id"], row["mode"]): row for row in old_rows}
    pairs = [(row, old[row["id"], row["mode"]]) for row in rows if (row["id"], row["mode"]) in old]
    return {"baseline_run_id": old_run["run_id"], "run_id": run["run_id"],
            "headline": {group: {"current": stats([row["correctness"] for row in rows if row["group"] == group]),
                                   "baseline": stats([row["correctness"] for row in old_rows if row["group"] == group]),
                                   "grounding_current": stats([row["grounding"] for row in rows if row["group"] == group]),
                                   "grounding_baseline": stats([row["grounding"] for row in old_rows if row["group"] == group])}
                         for group in sorted({row["group"] for row in [*rows, *old_rows]})},
            "paired_cohort": {"denominator": len(pairs), "current": stats([row["correctness"] for row, _ in pairs]),
                              "baseline": stats([row["correctness"] for _, row in pairs]),
                              "grounding_current": stats([row["grounding"] for row, previous in pairs if previous["grounding"] is not None and row["grounding"] is not None]),
                              "grounding_baseline": stats([previous["grounding"] for row, previous in pairs if previous["grounding"] is not None and row["grounding"] is not None])},
            "per_case": [{"id": row["id"], "mode": row["mode"], "correctness_delta": row["correctness"] - previous["correctness"],
                          "grounding_delta": row["grounding"] - previous["grounding"] if row["grounding"] is not None and previous["grounding"] is not None else None}
                         for row, previous in pairs]}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def aggregate(directory, baseline_dir=None):
    directory = Path(directory)
    if baseline_dir is not None and directory.resolve() == Path(baseline_dir).resolve():
        raise ValueError("run and baseline directories must differ")
    loaded = load(directory)
    summary = summarize(directory, loaded)
    comparison = compare(loaded, load(Path(baseline_dir))) if baseline_dir else None
    anchor_path = directory / "baseline_manifest.json"
    if baseline_dir and anchor_path.exists():
        anchor = read(anchor_path)
        if anchor.get("baseline_run_id") != comparison["baseline_run_id"]:
            raise ValueError("baseline anchor run ID mismatch")
        for name, digest in anchor.get("files", {}).items():
            if sha(Path(baseline_dir) / name) != digest:
                raise ValueError("immutable baseline artifact mismatch: " + name)
    with (directory / "answers.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "group", "mode", "status", "question", "reference", "answer", "correctness_completeness", "cited_fragment_grounding", "latency_seconds", "sources", "review_notes"])
        writer.writeheader()
        for row in loaded[1]:
            case = next(case for case in loaded[0]["scenarios"] if case["id"] == row["id"])
            latency = row.get("confirmed_latency_ms") if row["mode"] == "expanded" else row.get("wall_latency_ms")
            writer.writerow({**{key: row.get(key) for key in ("id", "group", "mode", "status")},
                             "question": case.get("question"), "reference": case.get("reference"), "answer": row.get("response"),
                             "correctness_completeness": row["correctness"], "cited_fragment_grounding": row["grounding"],
                             "latency_seconds": latency / 1000 if latency is not None else None,
                             "sources": json.dumps(row.get("sources"), ensure_ascii=False), "review_notes": row["notes"]})
    summary["artifact_sha256"]["answers.csv"] = sha(directory / "answers.csv")
    write_json(directory / "summary.json", summary)
    if comparison is not None:
        write_json(directory / "comparison.json", comparison)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args()
    aggregate(args.run_dir, args.baseline_dir)
