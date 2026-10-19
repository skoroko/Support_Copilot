"""Candidate-only ASGI entry point. No polling, notify server or production alias."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import Header, HTTPException

from app.core.config import PROJECT_ROOT, get_settings
from app.eval.candidate_identity import (
    candidate_identity,
    source_identity,
    validate_isolation,
)

LOADED_SOURCE_SHA256 = source_identity(PROJECT_ROOT)

if os.environ.get("S0005_CANDIDATE") != "true":
    raise RuntimeError("explicit isolated candidate opt-in required")
settings = get_settings()
validate_isolation(settings)
LOADED_IDENTITY = candidate_identity(settings, PROJECT_ROOT)
if not Path(os.environ["S0005_BUDGET_LEDGER"]).is_file():
    raise RuntimeError("shared acceptance budget ledger required before startup")

from app.eval.acceptance_budget import install_acceptance_budget

ledger = install_acceptance_budget(Path(os.environ["S0005_BUDGET_LEDGER"]), limit_usd=10.)
# Resolve the optional tokenizer before readiness, so vocabulary bootstrap is
# outside user-request latency. Its normal offline fallback remains available.
from app.chat.service import _o200k_base_encoding

TOKENIZER_WARM = _o200k_base_encoding() is not None
from app.main import app


@app.get("/__candidate_identity", include_in_schema=False)
async def identity(x_internal_token: str = Header(default="")) -> dict:
    if not secrets.compare_digest(x_internal_token, settings.internal_token.get_secret_value()):
        raise HTTPException(status_code=401, detail="unauthorized")
    payload = candidate_identity(settings, PROJECT_ROOT)
    if payload["candidate_identity_sha256"] != LOADED_IDENTITY["candidate_identity_sha256"]:
        raise HTTPException(status_code=409, detail="candidate source or artifacts changed; restart required")
    return {**payload, "loaded_source_sha256": LOADED_SOURCE_SHA256, "tokenizer_warm": TOKENIZER_WARM,
            "budget_enforced": True,
            "budget": ledger.summary()}
