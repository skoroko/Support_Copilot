"""Live replay of the fixed verification set for before/after measurements.

Runs each case once (or ``--repeats`` times) through the production agent
graph: a fresh ``InMemorySaver`` checkpoint per attempt, the configured
``AgentModel`` and the real ``RAGService``. Telegram delivery is forbidden
(read-only role plus a raising adapter) and no production checkpoints are
mutated.

Outputs:

- ``var/verification_<label>_private.json`` — full captures incl. answer text.
  Private working data; ``var/`` is gitignored.
- ``tests/eval/results/<label>/summary.json`` — content-free metrics only
  (timings, token usage, citation integrity, hashes). Safe to commit.

Usage (from ``HelpDesk_Assistant/``)::

    .venv/Scripts/python scripts/measure_verification_set.py --label baseline
    .venv/Scripts/python scripts/measure_verification_set.py --label after_block1 --ids golden-01,corpus-dev-019
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import logging
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SET_PATH = ROOT / "tests" / "eval" / "verification_set.json"


def _normalized_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _slot_hit(slot: dict, haystacks: list[str]) -> bool:
    quote = _normalized_text(slot.get("exact_quote"))
    if not quote:
        return False
    return any(quote in haystack for haystack in haystacks)


def _measure_row(row: dict, item: dict) -> dict:
    sources = row["answer"]["sources"]
    citations = [
        int(number)
        for number in re.findall(r"\[(\d+)\]", row["answer"]["answer"])
    ]
    records = {r["source"]["chunk_id"]: r for r in row["citation_evidence"]}
    evidence_texts = [_normalized_text(r["text"]) for r in row["citation_evidence"]]
    retrieved_texts = [
        _normalized_text(text)
        for retrieval in row["retrievals"]
        for text in (retrieval["result"].get("contexts") or [])
    ]
    slots = item.get("required_evidence_slots") or []
    slots_in_answer = sum(
        _slot_hit(slot, evidence_texts) for slot in slots
    )
    slots_in_retrieval = sum(
        _slot_hit(slot, retrieved_texts) for slot in slots
    )
    return {
        "question_id": row["id"],
        "group": item["group"],
        "expected_behavior": item["expected_behavior"],
        "repeat": row["repeat"],
        "total_seconds": row["total_seconds"],
        "model_seconds": sum(r["seconds"] for r in row["model_calls"]),
        "model_call_count": len(row["model_calls"]),
        "retrieval_seconds": sum(r["seconds"] for r in row["retrievals"]),
        "retrieval_count": len(row["retrievals"]),
        "selection_seconds": sum(row["selection_seconds"]),
        "input_tokens": [r["usage"]["input_tokens"] for r in row["model_calls"]],
        "output_tokens": [r["usage"]["output_tokens"] for r in row["model_calls"]],
        "reasoning_tokens": [
            ((r["usage"].get("output_token_details") or {}).get("reasoning"))
            for r in row["model_calls"]
        ],
        "knowledge_status": row.get("knowledge_status"),
        "translated_query_count": sum(
            int((retrieval["result"].get("telemetry") or {}).get("translated_query_count") or 0)
            for retrieval in row["retrievals"]
        ),
        "evidence_route": row.get("evidence_route"),
        "incomplete": row["answer"]["answer_info"]["incomplete"],
        "cited_source_count": len(sources),
        "answer_chars": len(row["answer"]["answer"]),
        "all_citations_resolve": bool(citations)
        and all(1 <= n <= len(sources) for n in citations),
        "all_cited_sources_have_exact_evidence": all(
            s["chunk_id"] in records and records[s["chunk_id"]]["text"]
            for s in sources
        ),
        "required_slots_total": len(slots),
        "required_slots_in_answer_evidence": slots_in_answer,
        "required_slots_in_retrieval": slots_in_retrieval,
        "answer_sha256": hashlib.sha256(row["answer"]["answer"].encode()).hexdigest(),
    }


async def _run(args: argparse.Namespace) -> dict:
    from app.core.config import get_settings
    from app.services.agent_persistent import (
        PERSISTENT_SYSTEM_PROMPT,
        TOOLS,
        build_agent,
    )
    from app.services.rag import RAGConfig, RAGService, build_llm_translate_hook
    from app.services.runtime_model import AgentModel
    from app.chat.agent import ChatAgent
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain_core.messages import HumanMessage, messages_to_dict

    payload = json.loads(Path(args.set).read_text(encoding="utf-8"))
    items = payload["items"]
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [item for item in items if item["id"] in wanted]
    if args.max_cases:
        items = items[: args.max_cases]

    settings = get_settings()
    rag_config = replace(RAGConfig.from_settings(settings), ingest_on_start=False)
    rag = RAGService(
        rag_config,
        translate_hook=(
            build_llm_translate_hook(rag_config)
            if getattr(rag_config, "translate_llm_enabled", True)
            else None
        ),
    ).build()

    calls: list[dict] = []
    retrieves: list[dict] = []
    stages: list[float] = []

    class TimedModel(AgentModel):
        async def ainvoke(self, *a, **kw):
            start = time.perf_counter()
            result = await super().ainvoke(*a, **kw)
            calls.append(
                {
                    "seconds": time.perf_counter() - start,
                    "usage": result.usage_metadata,
                    "tools": len(result.tool_calls),
                }
            )
            return result

    original_select = rag._select_context

    def selection(*a, **kw):
        start = time.perf_counter()
        try:
            return original_select(*a, **kw)
        finally:
            stages.append(time.perf_counter() - start)

    rag._select_context = selection
    original_retrieve = rag.retrieve

    def retrieve(*a, **kw):
        start = time.perf_counter()
        result = original_retrieve(*a, **kw)
        retrieves.append(
            {
                "seconds": time.perf_counter() - start,
                "args": [str(arg)[:200] for arg in a],
                "kwargs": {
                    key: (value if key not in {"history"} else f"<{len(value)} messages>")
                    for key, value in kw.items()
                },
                "result": result,
            }
        )
        return result

    rag.retrieve = retrieve
    model = TimedModel(settings, TOOLS)

    async def forbidden_send(*a, **kw):
        raise AssertionError("External send forbidden")

    context_budget = (
        settings.chat_model_context_tokens
        - settings.chat_response_tokens
        - settings.agent_reasoning_token_reserve
        - settings.chat_safety_margin_tokens
    )
    rows: list[dict] = []
    measured: list[dict] = []
    private_path = ROOT / "var" / f"verification_{args.label}_private.json"
    done_keys: set[tuple[str, int]] = set()
    if args.resume and private_path.exists():
        rows = json.loads(private_path.read_text(encoding="utf-8"))
        done_keys = {(row["id"], row["repeat"]) for row in rows}
        if done_keys:
            print(f"resuming: {len(done_keys)} capture(s) already done", file=sys.stderr)
    code_files = [
        "app/services/agent_persistent.py",
        "app/services/agent_react.py",
        "app/services/rag.py",
        "app/services/runtime_model.py",
        "app/services/prepared_evidence.py",
        "app/services/answer_quality.py",
        "app/services/evidence_text.py",
    ]
    code_sha256 = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in code_files
    }
    manifest = {
        "protocol": "verification-52 live replay; fresh in-memory graph per attempt; "
        "production AgentModel and RAG; sequential; no Telegram sends; "
        "no production checkpoint mutations",
        "set_id": payload["set_id"],
        "label": args.label,
        "runtime_model": settings.default_model,
        "reasoning_effort": settings.agent_reasoning_effort,
        "expanded_reasoning_effort": getattr(settings, "agent_expanded_reasoning_effort", None),
        "rag_collection": rag.config.collection_name,
        "response_tokens": settings.chat_response_tokens,
        "prompt_sha256": hashlib.sha256(PERSISTENT_SYSTEM_PROMPT.encode()).hexdigest(),
        "code_sha256": code_sha256,
        "case_count": len(items),
        "repeats": args.repeats,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        for item in items:
            for repeat in range(args.repeats):
                if (item["id"], repeat) in done_keys:
                    continue
                calls.clear()
                retrieves.clear()
                stages.clear()
                graph = build_agent(
                    InMemorySaver(),
                    model=model,
                    rag_provider=lambda: rag,
                    send_message=forbidden_send,
                )
                config = {
                    "configurable": {
                        "thread_id": f'{args.label}-{item["id"]}-{repeat}',
                        "user_role": "read-only",
                        "response_tokens": settings.chat_response_tokens,
                        "history_window": settings.chat_context_window,
                        "context_budget": context_budget,
                        "prepared_evidence_enabled": bool(
                            getattr(settings, "agent_prepared_evidence_enabled", True)
                        ),
                    }
                }
                start = time.perf_counter()
                # Transient provider/network errors must not kill a long run;
                # a persistent failure still surfaces after the last attempt.
                for attempt in range(3):
                    try:
                        state = await graph.ainvoke(
                            {"messages": [HumanMessage(content=item["question"])]}, config
                        )
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(5 * (attempt + 1))
                row = {
                    "id": item["id"],
                    "repeat": repeat,
                    "total_seconds": time.perf_counter() - start,
                    "model_calls": list(calls),
                    "retrievals": [
                        {**r, "result": {**r["result"], "sources": r["result"].get("sources")}}
                        for r in retrieves
                    ],
                    "selection_seconds": list(stages),
                    "answer": ChatAgent._result(state, None),
                    "citation_evidence": state.get("citation_evidence"),
                    "knowledge_status": state.get("knowledge_status"),
                    "evidence_route": state.get("evidence_route"),
                    "messages": messages_to_dict(state["messages"]),
                }
                rows.append(row)
                private_path.write_text(
                    json.dumps(rows, ensure_ascii=False, default=str), encoding="utf-8"
                )
                expanded_items = _expanded(items, args.repeats)
                position = {
                    (item["id"], repeat): index
                    for index, (item, repeat) in enumerate(
                        (item, repeat) for item in items for repeat in range(args.repeats)
                    )
                }
                # Rows captured by a wider earlier selection (e.g. a --resume
                # after a partial --ids run) keep their stored order at the end;
                # they cannot be re-measured without their suite items.
                stored_order = {id(row): index for index, row in enumerate(rows)}
                rows.sort(
                    key=lambda row: position.get(
                        (row["id"], row["repeat"]),
                        len(position) + stored_order[id(row)],
                    )
                )
                measured = [
                    _measure_row(row, expanded_items[position[(row["id"], row["repeat"])]])
                    for row in rows
                    if (row["id"], row["repeat"]) in position
                ]
                out_dir = ROOT / "tests" / "eval" / "results" / args.label
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "summary.json").write_text(
                    json.dumps(
                        {"manifest": manifest, "results": measured},
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
                    + "\n",
                    encoding="utf-8",
                )
    finally:
        rag.close()
    return {"manifest": manifest, "results": measured}


def _expanded(items: list[dict], repeats: int) -> list[dict]:
    return [item for item in items for _ in range(repeats)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Run label; used for var/ and results paths")
    parser.add_argument("--set", type=Path, default=DEFAULT_SET_PATH)
    parser.add_argument("--ids", default="", help="Comma-separated question ids to run")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Skip (id, repeat) pairs already present in the private capture")
    args = parser.parse_args()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        logging.disable(logging.CRITICAL)
        result = asyncio.run(_run(args))
    results = result["results"]
    totals = [r["total_seconds"] for r in results]
    print(
        json.dumps(
            {
                "label": args.label,
                "cases": len(results),
                "median_total_seconds": sorted(totals)[len(totals) // 2] if totals else None,
                "total_seconds": [round(t, 1) for t in totals],
                "model_calls": [r["model_call_count"] for r in results],
                "retrievals": [r["retrieval_count"] for r in results],
                "cited_sources": [r["cited_source_count"] for r in results],
                "slots_answer": [r["required_slots_in_answer_evidence"] for r in results],
                "output": f"tests/eval/results/{args.label}/summary.json",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
