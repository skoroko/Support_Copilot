"""Merge immutable same-candidate profile runs without re-executing adapters.

Original attempt payloads, failures and observed timings remain byte-for-byte
equivalent JSON values. A new binding records every parent file hash.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.eval.corpus_suite import digest, file_sha, read_json, report, write_new


def merge_runs(paths: list[Path], cases: list[dict], policy: dict) -> dict:
    if len(paths) < 2 or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("At least two distinct immutable parent runs required")
    runs = [read_json(p) for p in paths]
    invariant = ("schema_version", "split", "repeats", "implementation_sha256",
                 "suite_manifest_sha256", "frozen_context_sha256",
                 "machine_reference_contract_sha256", "retrieval_scoring_scope",
                 "candidate_identity", "selected_cases", "full_split_cases", "subset_only")
    first = runs[0]
    if not first.get("candidate_identity") or any(
            any(run.get(k) != first.get(k) for k in invariant) for run in runs[1:]):
        raise ValueError("Parent runs must bind identical candidate, corpus, references and selection")
    profiles, attempts, seen = {}, [], set()
    for run in runs:
        if set(profiles) & set(run["profiles"]):
            raise ValueError("Profile names must be disjoint; repetitions cannot be silently collapsed")
        profiles.update(run["profiles"])
        for row in run["attempts"]:
            if row["attempt_id"] in seen or row.get("semantic_review"):
                raise ValueError("Duplicate or previously reviewed attempt requires separate reconciliation")
            seen.add(row["attempt_id"])
            attempts.append(row)
    merged = {k: first[k] for k in invariant if k in first}
    merged.update(run_id=uuid4().hex, profiles=profiles, attempts=attempts,
                  legacy_regression_verified=False,
                  merge_method="immutable_parent_attempts_no_adapter_execution",
                  parent_runs=[{"run_id": run["run_id"], "binding_sha256": run["binding_sha256"],
                                "file_sha256": file_sha(path), "path": str(path)}
                               for run, path in zip(runs, paths, strict=True)])
    merged["binding_sha256"] = digest({k: v for k, v in merged.items() if k != "attempts"})
    merged["summary"] = report(merged, cases, policy)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = read_json(args.cases)
    merged = merge_runs(args.runs, cases, read_json(args.policy))
    write_new(args.output, merged)
    print(json.dumps({"parents": len(args.runs), "attempts": len(merged["attempts"]),
                      "profiles": list(merged["profiles"]), "provider_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
