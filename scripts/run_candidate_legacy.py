"""Run unchanged legacy 34+9+5 scenarios against an attested candidate only."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def run(output: Path, freeze_path: Path) -> dict:
    import httpx

    from app.core.config import get_settings
    from app.eval.candidate_identity import source_identity
    from scripts.evaluate_telegram_agent import run as legacy_run

    if output.exists():
        raise FileExistsError("Existing legacy attempts must not be replayed")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    settings = get_settings()
    async with httpx.AsyncClient(base_url="http://127.0.0.1:18001", timeout=10) as client:
        async def attest():
            response = await client.get("/__candidate_identity", headers={
                "X-Internal-Token": settings.internal_token.get_secret_value()})
            response.raise_for_status()
            identity = response.json()
            if (identity.get("candidate_identity_sha256") != freeze["candidate_identity_sha256"]
                    or identity.get("loaded_source_sha256") != source_identity(ROOT)
                    or not identity.get("budget_enforced")):
                raise ValueError("Frozen candidate identity/budget mismatch")
            return identity

        before = await attest()
        args = argparse.Namespace(api_url="http://127.0.0.1:18001", output=output,
            runtime_manifest=freeze_path, allow_external_data_transfer=True, limit=None,
            resume_safe=False, no_expansions=False, expansion_indices="1,10,16,24,32")
        payload = await legacy_run(args)
        after = await attest()
        payload.update(candidate_identity_sha256=freeze["candidate_identity_sha256"],
            budget_before=before["budget"], budget_after=after["budget"],
            native_polling_measured=False, actual_telegram_delivery_measured=False,
            independently_semantically_reviewed=False)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"run_id": payload["run_id"], "groups": payload["structural_proxies"]["groups"],
                "semantic_acceptance": "pending", "budget": after["budget"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--candidate-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-external-data-transfer", action="store_true")
    args = parser.parse_args()
    if not args.allow_external_data_transfer:
        parser.error("explicit source egress authorization required")
    logging.disable(logging.CRITICAL)
    try:
        print(json.dumps(asyncio.run(run(args.output, args.candidate_freeze))))
    except BaseException as exc:  # noqa: BLE001 - no provider/credential-bearing error text
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1) from None
