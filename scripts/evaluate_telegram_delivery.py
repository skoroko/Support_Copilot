"""Explicitly authorized real Telegram delivery probe against isolated candidate.

One invocation = one new attempt; existing output is never replayed. The budget
ledger is enforced by the candidate provider adapters, not by HTTP timing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secrets-file", required=True, type=Path)
    parser.add_argument("--api-url", default="http://127.0.0.1:18001")
    parser.add_argument("--question-file", required=True, type=Path)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recipient-fingerprint", required=True)
    parser.add_argument("--candidate-identity", required=True)
    parser.add_argument("--authorized-working-chat", action="store_true")
    parser.add_argument("--confirm-expanded", action="store_true")
    parser.add_argument("--no-documents-panel", action="store_true")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if not args.authorized_working_chat:
        parser.error("explicit working-chat authorization flag required")
    # Avoid library INFO records with Telegram chat IDs, question text or token URLs.
    logging.disable(logging.CRITICAL)
    from app.eval.telegram_delivery import run_delivery
    from bot.config import BotSettings
    try:
        settings = BotSettings(_env_file=args.secrets_file)
        question = args.question_file.read_text(encoding="utf-8").strip()
        if not question or question.startswith("/"):
            raise ValueError("non-command text question required")
        result = asyncio.run(run_delivery(
            settings=settings, api_url=args.api_url, question=question,
            scenario_id=args.scenario_id, recipient_fingerprint=args.recipient_fingerprint,
            expected_identity=args.candidate_identity, source_root=ROOT, output=args.output,
            timeout=args.timeout, expanded=args.confirm_expanded,
            with_sources=not args.no_documents_panel))
    except BaseException as exc:  # noqa: BLE001 - interruption and SDK text may contain secrets
        # Never print exception strings: provider/Telegram validation may embed credentials.
        print(json.dumps({"status": "failed_preflight", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({key: result[key] for key in (
        "status", "scenario_id", "candidate_identity_sha256", "recipient_sha256",
        "receipt_to_first_body_ack_ms", "receipt_to_last_answer_ack_ms",
        "receipt_to_last_delivery_ack_ms", "telegram_failures")}))
    return 0 if result["status"] == "delivered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
