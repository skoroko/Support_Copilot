"""Opt-in live benchmark for the homework 6.2 and 6.3 agent variants.

The module never sends an API request unless its CLI receives
``--allow-external-api``.  Importing helpers and running unit tests therefore
remain entirely offline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict

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
    from app.services.agent_react import run_react_agent

    started = time.perf_counter()
    # Keep main and critic on the same explicitly configured existing model.
    result = run_react_agent(case["task"], model_critic=os.environ["OPENAI_MODEL"], timeout_per_iteration_sec=15)
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
        "answer": result.get("answer", ""),
        "trace": result.get("trace", []),
        "usage": usage,
    }


async def _custom_record(case: dict[str, str], repeat: int, checkpoint: Callable = lambda _: None) -> dict[str, Any]:
    from app.services.agent_graph import MAX_ITERATIONS_MESSAGE, custom_graph

    started = time.perf_counter()
    state: dict[str, Any] = {}
    async for state in custom_graph.astream(
        {
            "messages": [HumanMessage(content=case["task"])],
            "iteration_count": 0,
            "tool_results": [],
        },
        config={"configurable": {"thread_id": f"bench-{case['id']}-{repeat}"}, "recursion_limit": 20},
        stream_mode="values",
    ):
        checkpoint(_state_snapshot(state))
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    snapshot = _state_snapshot(state)
    if snapshot["answer"] == MAX_ITERATIONS_MESSAGE:
        snapshot["error"] = "max_iterations=6"
    return {
        "case_id": case["id"],
        "implementation": "custom_langgraph",
        "repeat": repeat,
        "latency_ms": elapsed_ms,
        **usage_from_messages(state["messages"]),
        "total_steps": int(state.get("iteration_count", 0) or 0),
        "error": None,
        **snapshot,
    }


async def _prebuilt_record(case: dict[str, str], repeat: int, checkpoint: Callable = lambda _: None) -> dict[str, Any]:
    from app.services.agent_graph import prebuilt_graph

    started = time.perf_counter()
    state: dict[str, Any] = {}
    async for state in prebuilt_graph.astream(
        {"messages": [HumanMessage(content=case["task"])]},
        config={"recursion_limit": 20},
        stream_mode="values",
    ):
        checkpoint(_state_snapshot(state))
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
        **_state_snapshot(state),
    }


def _state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    from app.services.runtime_model import visible_text

    messages = state.get("messages", [])
    answers = [message for message in messages if isinstance(message, AIMessage)]
    metadata = answers[-1].response_metadata if answers else {}
    incomplete = metadata.get("status") == "incomplete" or metadata.get("finish_reason") in {"length", "content_filter", "error"}
    return {
        "answer": visible_text(answers[-1].content) if answers else "",
        "messages": messages_to_dict(messages),
        "tool_results": state.get("tool_results", []),
        **usage_from_messages(messages),
        "total_steps": int(state.get("iteration_count", len(answers))),
        "provider_status": metadata.get("status") or metadata.get("finish_reason"),
        "error": "provider generation incomplete" if incomplete else None,
    }


def _telegram_stub(chat_id: str, text: str) -> str:
    return f"Локальная имитация: chat_id={chat_id}; text={text}. Telegram API не вызывался."


def _worker(case: dict[str, str], repeat: int, implementation: str, connection: Any) -> None:
    """Isolate network calls and replace notification handlers before imports."""
    try:
        import app.services.agent_naive_tools as tools
        import app.services.agent_react_tools as react_tools

        tools.send_telegram_message = _telegram_stub
        react_tools.DISPATCH["send_telegram_message"] = _telegram_stub
        import app.services.agent_react as react
        original_execute = react._execute_tool_before_deadline

        def execute(tool_name: str, tool_args: dict[str, Any], deadline: float, **kwargs: Any) -> str:
            if tool_name == "send_telegram_message":
                return _telegram_stub(**tool_args)
            return original_execute(tool_name, tool_args, deadline, **kwargs)

        react._execute_tool_before_deadline = execute
        if os.environ.get("BENCH_API") == "responses":
            from scripts.benchmark_responses import ResponsesChatBridge
            original_openai = react.OpenAI
            react.OpenAI = lambda **kwargs: ResponsesChatBridge(
                original_openai(**kwargs), effort=os.environ["BENCH_REASONING"],
                service_tier=os.environ.get("BENCH_SERVICE_TIER") or None,
            )
        # Avoid unsupported fixed temperature on reasoning models. This is a
        # benchmark-only override and does not change the production factory.
        if implementation != "react_loop_6_2":
            import app.services.agent_graph as graph
            updates: dict[str, Any] = {"temperature": None, "store": False}
            if os.environ.get("BENCH_API") == "responses":
                updates.update(use_responses_api=True, output_version="v0", reasoning={"effort": os.environ["BENCH_REASONING"]},
                               include=["reasoning.encrypted_content"], max_tokens=8692)
            if os.environ.get("BENCH_SERVICE_TIER"):
                updates["service_tier"] = os.environ["BENCH_SERVICE_TIER"]
            graph.model = graph.model.model_copy(update=updates)
            graph.bound_model = graph.model.bind_tools(graph.TOOLS)
            graph.prebuilt_graph = graph.build_prebuilt_graph(graph.model)

        checkpoint = lambda snapshot: connection.send(("checkpoint", snapshot))
        if implementation == "react_loop_6_2":
            row = _baseline_record(case, repeat)
        elif implementation == "custom_langgraph":
            row = asyncio.run(_custom_record(case, repeat, checkpoint))
        else:
            row = asyncio.run(_prebuilt_record(case, repeat, checkpoint))
        connection.send(("result", row))
    except Exception as exc:  # noqa: BLE001 - every execution failure must remain a benchmark row
        # Do not persist provider exception bodies, which may contain request data.
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


def _isolated_record(case: dict[str, str], repeat: int, implementation: str, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "case_id": case["id"], "implementation": implementation, "repeat": repeat,
        "answer": "", "prompt_tokens": None, "completion_tokens": None,
        "total_tokens": None, "total_steps": None, "error": None,
    }
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(case, repeat, implementation, send))
    process.start()
    send.close()
    checkpoints: list[dict[str, Any]] = []
    deadline = started + timeout
    try:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or not receive.poll(max(0, remaining)):
                row["error"] = "execution timeout"
                break
            try:
                kind, value = receive.recv()
            except EOFError:
                row["error"] = "worker exited without result"
                break
            if kind == "checkpoint":
                checkpoints.append(value)
                row.update(value)
            elif kind == "result":
                row.update(value)
                break
            else:
                row["error"] = value
                break
    finally:
        receive.close()
        if process.is_alive():
            process.terminate()
        process.join()
    row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    row["checkpoints"] = checkpoints
    row["status"] = "failed" if row.get("error") else "completed"
    # Partial counts omit any provider response lost during timeout/failure.
    row["usage_complete"] = not bool(row.get("error"))
    return row


async def run_benchmark(
    cases: Iterable[dict[str, str]] = CASES,
    repeats: int = DEFAULT_REPEATS,
    *,
    execution_timeout: float = 180,
    on_row: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run 5 × 3 × 3 live calls sequentially, returning raw non-invented data."""

    if repeats < DEFAULT_REPEATS:
        raise ValueError(f"repeats must be at least {DEFAULT_REPEATS}")
    if execution_timeout <= 0:
        raise ValueError("execution_timeout must be positive")
    rows: list[dict[str, Any]] = []
    for case in cases:
        for repeat in range(1, repeats + 1):
            for implementation in IMPLEMENTATIONS:
                rows.append(await asyncio.to_thread(_isolated_record, case, repeat, implementation, execution_timeout))
                if on_row:
                    on_row(rows)
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
    parser.add_argument("--env-file", type=Path, help="Existing configuration file; never copied or printed.")
    parser.add_argument("--model", help="Existing OpenAI model; otherwise requires OPENAI_MODEL in configuration.")
    parser.add_argument("--execution-timeout", type=float, default=180)
    parser.add_argument("--api", choices=("chat_completions", "responses"), default="responses")
    parser.add_argument("--reasoning", choices=("none", "low", "medium", "high"), default="medium")
    parser.add_argument("--service-tier", choices=("auto", "default", "priority"), default="priority")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "agent-graph-benchmark.json")
    args = parser.parse_args()
    if not args.allow_external_api:
        raise SystemExit("External OpenAI calls are disabled. Re-run with --allow-external-api after approval.")

    from support_copilot.config import load_settings

    settings = load_settings(args.env_file) if args.env_file else load_settings()
    selected_model = args.model or settings.openai_model
    endpoint = urlsplit(settings.openai_base_url)
    if (endpoint.scheme != "https" or endpoint.hostname != "api.openai.com"
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise SystemExit("Benchmark permits only the existing official OpenAI endpoint.")
    if not settings.openai_api_key or not selected_model:
        raise SystemExit("Configured OPENAI_API_KEY and an explicit existing OPENAI_MODEL/--model are required.")
    # Spawned workers inherit only the selected provider settings. No fallback.
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    os.environ["OPENAI_BASE_URL"] = settings.openai_base_url
    os.environ["OPENAI_MODEL"] = selected_model
    os.environ["LLM_TIMEOUT_SEC"] = str(settings.llm_timeout_sec)
    os.environ["BENCH_API"] = args.api
    os.environ["BENCH_REASONING"] = args.reasoning
    os.environ["BENCH_SERVICE_TIER"] = args.service_tier
    payload = {
        "run_at": datetime.now(UTC).isoformat(),
        "repeats": args.repeats,
        "cases": list(CASES),
        "model": selected_model,
        "critic_model": selected_model,
        "api": args.api,
        "reasoning": args.reasoning if args.api == "responses" else None,
        "service_tier": args.service_tier,
        "react_iteration_timeout_seconds": 15,
        "temperature": "provider_default",
        "telegram": "local_stub_only",
        "knowledge_base": "support_copilot/data/knowledge_base.json (educational lexical KB)",
        "execution_timeout_seconds": args.execution_timeout,
        "expected_executions": len(CASES) * args.repeats * len(IMPLEMENTATIONS),
        "rows": [],
        "status": "running",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(rows: list[dict[str, Any]]) -> None:
        payload["rows"] = rows
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output)

    save([])
    rows = asyncio.run(run_benchmark(repeats=args.repeats, execution_timeout=args.execution_timeout, on_row=save))
    payload["status"] = "finished"
    payload["completed_executions"] = sum(row["status"] == "completed" for row in rows)
    payload["failed_executions"] = sum(row["status"] == "failed" for row in rows)
    save(rows)
    print(markdown_table(rows))
    print(args.output)


if __name__ == "__main__":
    main()
