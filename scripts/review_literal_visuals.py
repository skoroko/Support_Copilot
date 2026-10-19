"""Separate machine literal review of exact source occurrences; no expert claim."""
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
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument("--secrets-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--resume-unattempted", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Verify/render exact occurrences without a provider")
    parser.add_argument("--allow-external-data-transfer", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only and not args.allow_external_data_transfer:
        parser.error("Explicit corpus egress authorization required")
    if not args.prepare_only and (not args.budget_ledger or not args.secrets_file):
        parser.error("Existing task ledger and explicit secrets file required")
    if (args.output_dir / "review_attempts.sqlite3").exists() and not args.resume_unattempted:
        parser.error("Existing review run requires --resume-unattempted; attempted requests never replay")
    logging.disable(logging.CRITICAL)
    try:
        from app.eval.visual_literal_review import (
            bound_candidates,
            prepare_occurrence_crops,
            review_candidates,
        )
        from app.eval.visual_transcribe import (
            load_inventory,
            validate_official_endpoint,
        )
        if args.prepare_only:
            inventory = load_inventory(args.manifest)
            rows = bound_candidates(args.candidates, inventory)
            prepared = prepare_occurrence_crops(rows, inventory, manifest_path=args.manifest,
                data_root=args.data_dir, output_dir=args.output_dir / "occurrence_crops")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "prepared_review_candidates.json").write_text(
                json.dumps({"schema_version": 1, "records": prepared}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"mode": "offline_preparation", "prepared_occurrences": len(prepared),
                              "provider_calls": 0, "approved_occurrences": 0}))
            return 0
        from openai import OpenAI

        from app.core.config import Settings
        from app.eval.acceptance_budget import open_budget
        if not args.budget_ledger.is_file():
            raise ValueError("Existing shared ten dollar task ledger required")
        settings = Settings(_env_file=args.secrets_file)
        endpoint = validate_official_endpoint(settings.openai_base_url)
        ledger = open_budget(args.budget_ledger, limit_usd=10.)
        with OpenAI(api_key=settings.openai_api_key.get_secret_value(), base_url=endpoint,
                    timeout=120, max_retries=0) as client:
            result = review_candidates(manifest_path=args.manifest, candidates_path=args.candidates,
                data_root=args.data_dir, output_dir=args.output_dir, ledger=ledger, client=client,
                batch_size=args.batch_size, max_batches=args.max_batches)
        print(json.dumps(result, ensure_ascii=False))
        return 1 if result["failed_or_unknown_batches"] else 0
    except BaseException as exc:  # noqa: BLE001 - never print secrets/SDK exception bodies
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
