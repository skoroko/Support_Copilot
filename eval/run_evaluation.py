from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.parsing import parse_json_response  # noqa: E402
from app.services.prompting import build_support_messages  # noqa: E402

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - checked at runtime for real eval
    AsyncOpenAI = None  # type: ignore[assignment]


DEFAULT_GOLDEN = ROOT / "eval" / "golden_dataset.json"
DEFAULT_RUNS_DIR = ROOT / "eval" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Support Copilot golden dataset evaluation.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--judge", default=os.getenv("EVAL_JUDGE_MODEL"))
    parser.add_argument("--model-under-test", default=os.getenv("APP_DEFAULT_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--app-url", default=os.getenv("EVAL_APP_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=float(os.getenv("EVAL_TIMEOUT_SECONDS", "60")))
    parser.add_argument(
        "--offline-smoke",
        action="store_true",
        help="Use expected answers and a deterministic keyword judge; no API keys or network.",
    )
    args = parser.parse_args()
    if not args.offline_smoke and not args.judge:
        parser.error("--judge is required unless --offline-smoke is used")
    return args


def load_golden(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Golden dataset must be a JSON object")
    if payload.get("version") is None:
        raise ValueError("Golden dataset must contain top-level version")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Golden dataset must contain non-empty items list")
    return payload


async def run_real_evaluation(args: argparse.Namespace, golden: dict[str, Any]) -> dict[str, Any]:
    if AsyncOpenAI is None:
        raise RuntimeError("OpenAI SDK is not installed. Run `uv sync --locked --extra eval` first.")

    judge_client = AsyncOpenAI()
    async with httpx.AsyncClient(base_url=args.app_url, timeout=args.timeout) as app_client:
        items = []
        for item in golden["items"]:
            answer = await call_app(app_client, item, args.model_under_test)
            judge_result = await call_judge(judge_client, args.judge, item, answer)
            items.append(build_run_item(item, answer, judge_result))
    return build_run_payload(args.model_under_test, args.judge, golden["version"], items)


async def call_app(client: httpx.AsyncClient, item: dict[str, Any], model: str) -> str:
    response = await client.post(
        "/chat",
        json={
            "messages": build_support_messages(item["question"]),
            "model": model,
            "temperature": 0,
            "max_tokens": 600,
        },
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("content", ""))


async def call_judge(
    client: Any,
    judge_model: str,
    item: dict[str, Any],
    answer: str,
) -> dict[str, Any]:
    response = await client.chat.completions.create(
        model=judge_model,
        messages=build_judge_messages(item, answer),
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return normalize_judge_result(parse_json_response(content or "{}"))


def build_judge_messages(item: dict[str, Any], answer: str) -> list[dict[str, str]]:
    system = (
        "You are a strict LLM evaluation judge for a Russian technical support assistant. "
        "Use G-Eval style reason-then-score. Return only valid JSON. "
        "The JSON fields must appear in this order: reasoning, scores, explanation. "
        "Scores are integers from 1 to 5 for relevance, correctness, and completeness."
    )
    user = {
        "evaluation_steps": [
            "Compare the candidate answer with the expected answer and expected keywords.",
            "Check whether prohibited must_not_contain phrases or unsafe claims appear.",
            "Assess relevance to the user question.",
            "Assess factual correctness without rewarding invented MIDAS facts.",
            "Assess completeness for the support workflow.",
        ],
        "required_json_shape": {
            "reasoning": "Brief reason first, before any score fields.",
            "scores": {"relevance": 1, "correctness": 1, "completeness": 1},
            "explanation": "One-line summary.",
        },
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "expected_keywords": item.get("expected_keywords", []),
        "must_not_contain": item.get("must_not_contain", []),
        "candidate_answer": answer,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def normalize_judge_result(payload: dict[str, Any]) -> dict[str, Any]:
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("Judge JSON must contain scores object")

    normalized_scores: dict[str, int] = {}
    for name in ("relevance", "correctness", "completeness"):
        value = int(scores.get(name, 0))
        if value < 1 or value > 5:
            raise ValueError(f"Judge score {name} must be between 1 and 5")
        normalized_scores[name] = value

    return {
        "reasoning": str(payload.get("reasoning", "")),
        "scores": normalized_scores,
        "explanation": str(payload.get("explanation", "")),
    }


def run_offline_smoke(args: argparse.Namespace, golden: dict[str, Any]) -> dict[str, Any]:
    items = []
    for item in golden["items"]:
        answer = item["expected_answer"]
        judge_result = offline_keyword_judge(item, answer)
        items.append(build_run_item(item, answer, judge_result))
    return build_run_payload(
        args.model_under_test,
        args.judge or "offline-keyword-judge",
        golden["version"],
        items,
    )


def offline_keyword_judge(item: dict[str, Any], answer: str) -> dict[str, Any]:
    lowered = answer.casefold()
    keywords = [str(keyword).casefold() for keyword in item.get("expected_keywords", [])]
    forbidden = [str(term).casefold() for term in item.get("must_not_contain", [])]
    keyword_hits = sum(1 for keyword in keywords if keyword in lowered)
    forbidden_hits = sum(1 for term in forbidden if term and term in lowered)

    if forbidden_hits:
        correctness = 2
    elif not keywords:
        correctness = 4
    else:
        correctness = 5 if keyword_hits == len(keywords) else max(3, 2 + keyword_hits)

    score = min(5, correctness)
    return {
        "reasoning": "Offline smoke judge compares expected keywords and prohibited phrases only.",
        "scores": {"relevance": score, "correctness": correctness, "completeness": score},
        "explanation": f"{keyword_hits}/{len(keywords)} keywords matched; forbidden={forbidden_hits}.",
    }


def build_run_item(item: dict[str, Any], answer: str, judge_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "question": item["question"],
        "answer": answer,
        "scores": judge_result["scores"],
        "reasoning": judge_result["reasoning"],
        "explanation": judge_result["explanation"],
    }


def build_run_payload(
    model_under_test: str,
    judge_model: str,
    golden_version: int,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "run_id": f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        "timestamp": now.isoformat().replace("+00:00", "Z"),
        "model_under_test": model_under_test,
        "judge_model": judge_model,
        "golden_version": golden_version,
        "items": items,
        "aggregates": calculate_aggregates(items),
    }


def calculate_aggregates(items: list[dict[str, Any]]) -> dict[str, float]:
    if not items:
        raise ValueError("Cannot calculate aggregates for empty evaluation run")

    relevance = [item["scores"]["relevance"] for item in items]
    correctness = [item["scores"]["correctness"] for item in items]
    completeness = [item["scores"]["completeness"] for item in items]
    return {
        "relevance_avg": round(sum(relevance) / len(relevance), 3),
        "correctness_avg": round(sum(correctness) / len(correctness), 3),
        "completeness_avg": round(sum(completeness) / len(completeness), 3),
        "min_correctness": min(correctness),
    }


def default_out_path() -> Path:
    return DEFAULT_RUNS_DIR / f"{datetime.now(timezone.utc).date().isoformat()}.json"


def write_run(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def async_main() -> None:
    args = parse_args()
    golden = load_golden(args.golden)
    payload = run_offline_smoke(args, golden) if args.offline_smoke else await run_real_evaluation(args, golden)
    out = args.out or default_out_path()
    write_run(out, payload)
    print(json.dumps({"out": str(out), "aggregates": payload["aggregates"]}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(async_main())
