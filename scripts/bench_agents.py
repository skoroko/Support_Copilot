"""Opt-in live benchmark for the homework 6.2 and 6.3 agent variants.

The module never sends an API request unless its CLI receives
``--allow-external-api``.  Importing helpers and running unit tests therefore
remain entirely offline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage

from app.services.agent_graph import custom_graph, prebuilt_graph
from app.services.agent_react import run_react_agent
from scripts.run_agent_comparison import CASES


IMPLEMENTATIONS = ("react_loop_6_2", "custom_langgraph", "prebuilt_langchain")
DEFAULT_REPEATS = 3


def _usage_from_message(message: Any) -> dict[str, int]:
    """Normalize LangChain/OpenAI usage metadata without provider-specific casts."""

    usage = getattr(message, "usage_metadata", None) or {}
    response_usage = (getattr(message, "response_metadata", None) or {}).get("token_usage", {})
    prompt = int(usage.get("input_tokens", usage.get("prompt_tokens", response_usage.get("prompt_tokens", 0))) or 0)
    completion = int(usage.get("output_tokens", usage.get("completion_tokens", response_usage.get("completion_tokens", 0))) or 0)
    total = int(usage.get("total_tokens", response_usage.get("total_tokens", prompt + completion)) or (prompt + completion))
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def usage_from_messages(messages: Iterable[Any]) -> dict[str, int]:
    """Sum token usage from every AI message retained by a graph invocation."""

    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        current = _usage_from_message(message)
        for key, value in current.items():
            total[key] += value
    return total


def _baseline_record(case: dict[str, str], repeat: int) -> dict[str, Any]:
    started = time.perf_counter()
    result = run_react_agent(case["task"])
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    usage = result.get("usage", {})
    return {
        "case_id": case["id"],
        "implementation": "react_loop_6_2",
        "repeat": repeat,
        "latency_ms": elapsed_ms,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "total_steps": int(result.get("steps", 0) or 0),
        "error": result.get("error"),
    }


async def _custom_record(case: dict[str, str], repeat: int) -> dict[str, Any]:
    started = time.perf_counter()
    state = await custom_graph.ainvoke(
        {
            "messages": [HumanMessage(content=case["task"])],
            "iteration_count": 0,
            "tool_results": [],
        },
        config={"configurable": {"thread_id": f"bench-{case['id']}-{repeat}"}, "recursion_limit": 20},
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "case_id": case["id"],
        "implementation": "custom_langgraph",
        "repeat": repeat,
        "latency_ms": elapsed_ms,
        **usage_from_messages(state["messages"]),
        "total_steps": int(state.get("iteration_count", 0) or 0),
        "error": None,
    }


async def _prebuilt_record(case: dict[str, str], repeat: int) -> dict[str, Any]:
    started = time.perf_counter()
    state = await prebuilt_graph.ainvoke(
        {"messages": [HumanMessage(content=case["task"])]},
        config={"recursion_limit": 20},
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    messages = state["messages"]
    return {
        "case_id": case["id"],
        "implementation": "prebuilt_langchain",
        "repeat": repeat,
        "latency_ms": elapsed_ms,
        **usage_from_messages(messages),
        "total_steps": sum(isinstance(message, AIMessage) for message in messages),
        "error": None,
    }


async def run_benchmark(cases: Iterable[dict[str, str]] = CASES, repeats: int = DEFAULT_REPEATS) -> list[dict[str, Any]]:
    """Run 5 × 3 × 3 live calls sequentially, returning raw non-invented data."""

    if repeats < DEFAULT_REPEATS:
        raise ValueError(f"repeats must be at least {DEFAULT_REPEATS}")
    rows: list[dict[str, Any]] = []
    for case in cases:
        for repeat in range(1, repeats + 1):
            rows.append(await asyncio.to_thread(_baseline_record, case, repeat))
            rows.append(await _custom_record(case, repeat))
            rows.append(await _prebuilt_record(case, repeat))
    return rows


def markdown_table(rows: Iterable[dict[str, Any]]) -> str:
    """Render concrete per-run facts for pasting into the homework report."""

    lines = [
        "| case | implementation | repeat | latency ms | prompt tokens | completion tokens | total tokens | total steps | error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {implementation} | {repeat} | {latency_ms} | {prompt_tokens} | {completion_tokens} | {total_tokens} | {total_steps} | {error} |".format(
                **{**row, "error": row.get("error") or ""}
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark homework 6.2 and 6.3 agents on the same five cases")
    parser.add_argument(
        "--allow-external-api",
        action="store_true",
        help="Required: this performs at least 45 agent executions (provider requests can be higher).",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "agent-graph-benchmark.json")
    args = parser.parse_args()
    if not args.allow_external_api:
        raise SystemExit("External OpenAI calls are disabled. Re-run with --allow-external-api after approval.")

    rows = asyncio.run(run_benchmark(repeats=args.repeats))
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "repeats": args.repeats,
        "cases": list(CASES),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(markdown_table(rows))
    print(args.output)


if __name__ == "__main__":
    main()
