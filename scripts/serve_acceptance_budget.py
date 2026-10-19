"""Run the single local Windows owner of the existing $10 acceptance ledger."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    if not args.ledger.is_file():
        raise ValueError("Existing shared task ledger required")
    from app.core.config import get_settings
    from app.eval.acceptance_budget import BudgetLedger
    from app.eval.budget_rpc import ledger_server
    ledger = BudgetLedger(args.ledger, limit_usd=10.)
    settings = get_settings()
    with ledger_server(ledger, settings.internal_token.get_secret_value()) as server:
        print(json.dumps({"status": "ready", "port": 16334, "budget": ledger.reconcile_usage()}), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - never print credentials or connection error strings
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1) from None
