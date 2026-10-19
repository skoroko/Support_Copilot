"""Offline known-failure plan or strict OCR candidate coverage; no credentials or provider calls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument("--attempt", nargs=2, action="append", metavar=("JOURNAL", "ATTEMPT_ID"))
    parser.add_argument("--unknown-transport-attempt", nargs=2, action="append", metavar=("JOURNAL", "ATTEMPT_ID"))
    parser.add_argument("--data-dir", type=Path, help="original permitted corpus root for terminal transport source checks")
    parser.add_argument("--acknowledge-held-parent-reservations", action="store_true")
    parser.add_argument("--coverage-journal", type=Path, action="append")
    parser.add_argument("--approved-transcriptions", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(4, 8), default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sum(bool(value) for value in (args.attempt, args.unknown_transport_attempt, args.coverage_journal)) != 1:
        parser.error("choose known parent attempts OR terminal transport parents OR coverage journals")
    if (args.attempt or args.unknown_transport_attempt) and not args.budget_ledger:
        parser.error("known-failure planning requires existing shared budget ledger")
    if args.unknown_transport_attempt and (not args.data_dir or not args.acknowledge_held_parent_reservations):
        parser.error("terminal transport planning requires original --data-dir and explicit full held-charge acknowledgment")
    try:
        from app.eval.visual_failure_recovery import (
            build_recovery_plan,
            build_unknown_transport_plan,
            coverage_report,
            write_recovery_plan,
        )

        if args.attempt or args.unknown_transport_attempt:
            if args.unknown_transport_attempt:
                result = build_unknown_transport_plan(manifest_path=args.manifest, ledger_path=args.budget_ledger,
                    data_root=args.data_dir, attempts=[(Path(path), attempt_id) for path, attempt_id in args.unknown_transport_attempt],
                    batch_size=args.batch_size, acknowledge_held_parent_reservations=args.acknowledge_held_parent_reservations)
            else:
                result = build_recovery_plan(manifest_path=args.manifest, ledger_path=args.budget_ledger,
                    attempts=[(Path(path), attempt_id) for path, attempt_id in args.attempt], batch_size=args.batch_size)
            summary = {"plan_sha256": result["plan_sha256"], "parents": len(result["parents"]),
                "groups": len(result["groups"]), "occurrences": sum(len(row["region_ids"]) for row in result["groups"]),
                "batch_size": args.batch_size, "approval_status": "pending"}
        else:
            result = coverage_report(manifest_path=args.manifest, journals=args.coverage_journal,
                                     approved_path=args.approved_transcriptions)
            summary = result["summary"]
        write_recovery_plan(args.output, result)
        print(json.dumps(summary))
        return 0
    except Exception as exc:  # noqa: BLE001 - no raw provider/manual content in console
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
