"""Explicitly opt-in budgeted vision candidates; never auto-approve evidence."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contact-manifest", type=Path)
    parser.add_argument("--mode", choices=("full", "triage"), default="full")
    parser.add_argument("--triage-results", type=Path)
    parser.add_argument("--prior-triage-results", type=Path,
                        help="explicit new triage run: carry bound classifications and skip fully classified sheets")
    parser.add_argument("--exclude-reviewed-transcriptions", type=Path,
                        help="skip only exact individually approved occurrences (full mode)")
    parser.add_argument("--resolves-attempt-journal", type=Path,
                        help="read-only old journal for explicit missing-group recovery in a new full run")
    parser.add_argument("--resolves-attempt-id")
    parser.add_argument("--failed-known-recovery-plan", type=Path,
                        help="explicit immutable offline plan; new paid calls in a separate run, no unknown-call replay")
    parser.add_argument("--unknown-transport-new-attempt-plan", type=Path,
                        help="explicit NEW paid calls after terminal APIConnectionError; old full charge stays held")
    parser.add_argument("--acknowledge-held-parent-reservations", action="store_true",
                        help="acknowledge unknown prior calls may have been billed; never release/reuse their reservations")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--secrets-file", type=Path, required=True)
    parser.add_argument("--allow-external-data-transfer", action="store_true")
    parser.add_argument("--resume-unattempted", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-count", type=int, default=1,
                        help="full mode only: split the fixed global batch plan across 1..4 independent runs")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--crop", choices=("full", "classifier"), default="full")
    parser.add_argument("--detail", choices=("low", "high", "auto"), default="high")
    parser.add_argument("--reasoning", choices=("none", "low"), default="none")
    parser.add_argument("--max-output-tokens", type=int, choices=(4096, 8192))
    args = parser.parse_args()
    if not args.allow_external_data_transfer:
        parser.error("explicit corpus egress authorization flag required")
    if args.mode == "triage" and args.contact_manifest is None:
        parser.error("triage requires --contact-manifest")
    if (args.output_dir / "attempts.sqlite3").exists() and not args.resume_unattempted:
        parser.error("existing run requires --resume-unattempted; attempted calls are never replayed")
    logging.disable(logging.CRITICAL)
    try:
        from openai import OpenAI

        from app.core.config import Settings
        from app.eval.acceptance_budget import BudgetLedger
        from app.eval.visual_transcribe import (
            transcribe_inventory,
            validate_official_endpoint,
        )
        settings = Settings(_env_file=args.secrets_file)
        endpoint = validate_official_endpoint(settings.openai_base_url)
        if not args.budget_ledger.is_file():
            raise ValueError("existing shared task budget ledger required; this CLI never creates a new task budget")
        ledger = BudgetLedger(args.budget_ledger, limit_usd=10.)
        with OpenAI(api_key=settings.openai_api_key.get_secret_value(), base_url=endpoint,
                    timeout=120, max_retries=0) as client:
            summary = transcribe_inventory(manifest_path=args.manifest, output_dir=args.output_dir,
                ledger=ledger, client=client, model="gpt-6-luna", catalog_path=args.catalog,
                batch_size=args.batch_size, max_batches=args.max_batches, crop=args.crop,
                detail=args.detail, effort=args.reasoning,
                output_tokens=args.max_output_tokens or (4096 if args.mode == "triage" else 8192),
                mode=args.mode, contact_path=args.contact_manifest, triage_results_path=args.triage_results,
                exclude_reviewed_transcriptions=args.exclude_reviewed_transcriptions,
                resolves_attempt_journal=args.resolves_attempt_journal, resolves_attempt_id=args.resolves_attempt_id,
                prior_triage_results=args.prior_triage_results,
                shard_count=args.shard_count, shard_index=args.shard_index,
                failed_known_recovery_plan=args.failed_known_recovery_plan,
                unknown_transport_new_attempt_plan=args.unknown_transport_new_attempt_plan,
                acknowledge_held_parent_reservations=args.acknowledge_held_parent_reservations)
        print(json.dumps(summary, ensure_ascii=False))
        return 1 if summary["states"].get("failed_or_unknown") else 0
    except BaseException as exc:  # noqa: BLE001 - credentials and SDK errors must never reach output
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
