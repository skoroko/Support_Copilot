"""Opt-in citation-presence judging of at most 34 saved educational answers.

Only the saved response is sent. No retrieval, answer generation, embedding,
context transfer or rescore of historical metric tables occurs.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.metrics import make_has_citation, make_has_citation_judge
from scripts.benchmark_responses import ResponsesStructuredJudge


def load_answers(path: Path, limit: int) -> list[str]:
    if not 1 <= limit <= 34:
        raise ValueError("limit must be between 1 and 34")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))[:limit]
    if not rows or any(not row.get("response", "").strip() for row in rows):
        raise ValueError("Every selected row must contain a saved response")
    return [row["response"] for row in rows]


async def evaluate(answers: list[str], judge: Any, *, on_row: Any) -> list[dict[str, Any]]:
    metric, regex = make_has_citation_judge(judge), make_has_citation(None)
    rows = []
    for index, answer in enumerate(answers, start=1):
        started = time.perf_counter()
        result = await metric.ascore(response=answer)
        original = await regex.ascore(response=answer)
        value = result.value if result.value in {"yes", "no"} else None
        rows.append({
            "row_number": index,
            "response_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "has_citation_judge": value,
            "has_citation_regex": original.value,
            "agreement": value == original.value if value is not None else None,
            "status": "completed" if value is not None else "failed",
            "error": getattr(judge, "last_error", None) if value is None else None,
            "usage": getattr(judge, "last_usage", None),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        on_row(rows)
    return rows


class RecordingJudge(ResponsesStructuredJudge):
    def __init__(self, client: Any):
        self.sdk_client = client
        self.last_error, self.last_usage = None, None
        super().__init__(SimpleNamespace(responses=SimpleNamespace(parse=self.parse)),
                         model="gpt-6-luna", effort="medium", service_tier="priority")

    async def parse(self, **kwargs):
        self.last_error, self.last_usage = None, None
        try:
            response = await self.sdk_client.responses.parse(**kwargs)
        except Exception as exc:
            self.last_error = {"type": type(exc).__name__, "status_code": getattr(exc, "status_code", None)}
            raise
        usage = response.usage
        if usage:
            self.last_usage = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "total_tokens": usage.total_tokens}
        if response.status != "completed" or response.output_parsed is None:
            self.last_error = {"type": "IncompleteStructuredVerdict", "response_status": response.status}
        return response


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from openai import AsyncOpenAI

    from support_copilot.config import load_settings

    settings = load_settings(args.env_file)
    endpoint = urlsplit(settings.openai_base_url)
    if endpoint.scheme != "https" or endpoint.hostname != "api.openai.com" or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("Only the existing official OpenAI endpoint is permitted")
    if not settings.openai_api_key:
        raise ValueError("Existing OpenAI API configuration is required")
    answers = load_answers(args.input, args.limit)
    payload = {
        "started_at": datetime.now(UTC).isoformat(),
        "source_file": args.input.name,
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "scope": "Citation presence only; no source verification, grounding or correctness assessment",
        "model": "gpt-6-luna", "api": "responses", "reasoning": "medium", "service_tier": "priority",
        "max_retries": 0, "timeout_seconds": 180, "raw_contexts_transferred": False,
        "planned_answers": len(answers), "rows": [], "status": "running",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(rows):
        payload["rows"] = rows
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output)

    save([])
    async with AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url, timeout=180, max_retries=0) as client:
        rows = await evaluate(answers, RecordingJudge(client), on_row=save)
    completed = [row for row in rows if row["status"] == "completed"]
    payload.update(status="finished", finished_at=datetime.now(UTC).isoformat(),
                   completed=len(completed), failed=len(rows) - len(completed),
                   yes=sum(row["has_citation_judge"] == "yes" for row in completed),
                   no=sum(row["has_citation_judge"] == "no" for row in completed),
                   agreements=sum(row["agreement"] is True for row in completed))
    save(rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=34)
    parser.add_argument("--allow-external-api", action="store_true")
    args = parser.parse_args()
    if not args.allow_external_api:
        raise SystemExit("External calls disabled; explicit --allow-external-api required")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    result = asyncio.run(run(args))
    print(json.dumps({key: result[key] for key in ("status", "completed", "failed", "yes", "no", "agreements")}))


if __name__ == "__main__":
    main()
