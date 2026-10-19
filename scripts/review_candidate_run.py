"""Independent machine-only semantic review; never expert approval or acceptance."""
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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--secrets-file", type=Path, required=True)
    parser.add_argument("--allow-external-data-transfer", action="store_true")
    parser.add_argument("--resume-unattempted", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--answers-per-batch", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--reasoning", choices=("low",), default="low")
    args = parser.parse_args()
    if not args.allow_external_data_transfer:
        parser.error("explicit source/answer egress authorization flag required")
    if (args.output_dir / "review_attempts.sqlite3").exists() and not args.resume_unattempted:
        parser.error("existing machine review requires --resume-unattempted; attempted calls are never replayed")
    logging.disable(logging.CRITICAL)
    try:
        from openai import OpenAI

        from app.core.config import Settings
        from app.eval.acceptance_budget import BudgetLedger
        from app.eval.semantic_review import review_run
        from app.eval.visual_transcribe import validate_official_endpoint
        if not args.budget_ledger.is_file():
            raise ValueError("existing shared task ledger required")
        settings = Settings(_env_file=args.secrets_file)
        endpoint = validate_official_endpoint(settings.openai_base_url)
        ledger = BudgetLedger(args.budget_ledger, limit_usd=10.)
        with OpenAI(api_key=settings.openai_api_key.get_secret_value(), base_url=endpoint,
                    max_retries=0, timeout=120) as client:
            artifact = review_run(run_path=args.run, cases_path=args.cases, contexts_path=args.contexts,
                contract_path=args.reference_contract, output_dir=args.output_dir, ledger=ledger, client=client,
                max_cases=args.max_cases, answers_per_batch=args.answers_per_batch, effort=args.reasoning)
        print(json.dumps({"summary": artifact["summary"], "budget": artifact["budget"]}, ensure_ascii=False))
        return int(any(state != "machine_review_completed" for state in artifact["review_attempt_states"]))
    except BaseException as exc:  # noqa: BLE001 - never print SDK errors, which may embed credentials
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
