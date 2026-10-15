"""Safely probe generation support; never print credentials or provider bodies."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import AsyncOpenAI

from app.core.config import get_settings


async def probe() -> dict:
    settings = get_settings()
    results = []
    async with AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        timeout=30,
        max_retries=0,
    ) as client:
        for model in ("gpt-5.4-mini", "gpt-6-luna"):
            try:
                response = await client.responses.create(
                    model=model,
                    input="Ответь только числом: 2 + 2.",
                    reasoning={"effort": "none"},
                    max_output_tokens=128,
                    store=False,
                )
                results.append(
                    {
                        "model": model,
                        "status": response.status,
                        "answer_correct": response.output_text.strip() == "4",
                        "input_tokens": response.usage.input_tokens
                        if response.usage
                        else None,
                        "output_tokens": response.usage.output_tokens
                        if response.usage
                        else None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - never print provider bodies
                results.append(
                    {
                        "model": model,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "http_status": getattr(exc, "status_code", None),
                    }
                )
    return {"endpoint_configured": bool(settings.openai_base_url), "probes": results}


if __name__ == "__main__":
    result = asyncio.run(probe())
    output = ROOT / "var/luna_migration/preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
