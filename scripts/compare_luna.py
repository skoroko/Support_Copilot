"""Compare runtime models through the real graph with fixed recorded evidence.

Gold references are scoring-only. Retrieval is frozen, Telegram delivery disabled.
Every paid call is recorded before dispatch; resume never repeats a saved attempt.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.chat.agent import ChatAgent
from app.core.config import get_settings
from app.services.agent_persistent import PERSISTENT_SYSTEM_PROMPT, TOOLS, build_agent
from app.services.runtime_model import AgentModel

VARIANTS = {
    "mini": ("gpt-5.4-mini", "none", "chat_completions"),
    **{
        f"luna-{effort}": ("gpt-6-luna", effort, "responses")
        for effort in ("none", "low", "medium", "high", "xhigh")
    },
}
# USD per 1M Standard short-context tokens; verified 2026-10-15.
# https://developers.openai.com/api/docs/pricing
RATES = {
    "gpt-5.4-mini": (0.75, 0.075, 0.75, 4.5),
    "gpt-6-luna": (0.1, 0.01, 0.125, 0.5),
    "gpt-6.1-sol": (2, 0.1, 2.5, 10),
}


def atomic_json(path: Path, payload: dict) -> None:
    """A Windows reader may briefly hold a file without delete sharing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)


def usage_cost(model: str, usage: dict) -> float:
    if not usage or "input_tokens" not in usage or "output_tokens" not in usage:
        raise ValueError("Missing usage; cost cannot be treated as zero")
    inp, cache, write, out = RATES[model]
    details = usage.get("input_token_details") or {}
    read_tokens, write_tokens = (
        int(details.get("cache_read") or 0),
        int(details.get("cache_creation") or 0),
    )
    input_tokens = int(usage["input_tokens"])
    if input_tokens > 272000:
        if model == "gpt-5.4-mini":
            raise ValueError("Input exceeds mini supported limit")
        inp, cache, write, out = inp * 2, cache * 2, write * 2, out * 1.5
    # output_tokens already includes reasoning. Never add reasoning a second time.
    return (
        max(0, input_tokens - read_tokens - write_tokens) * inp
        + read_tokens * cache
        + write_tokens * write
        + int(usage["output_tokens"]) * out
    ) / 1e6


def extract_cases(run: dict, checkpoints: dict) -> list[dict]:
    evidence = {}
    for thread in checkpoints["threads"].values():
        groups = []
        for message in thread.get("messages", []):
            if message["type"] == "human":
                groups.append([])
            elif groups:
                groups[-1].append(message)
        for ident, messages in zip(thread["scenario_ids"], groups):
            records, statuses = {}, []
            for message in messages:
                if (
                    message["type"] != "tool"
                    or message.get("name") != "search_knowledge_base"
                ):
                    continue
                payload = json.loads(message["content"])
                statuses.append(payload.get("status", "unavailable"))
                for record in payload.get("evidence", []):
                    number = record["citation_number"]
                    if number not in records or len(record["text"]) >= len(
                        records[number]["text"]
                    ):
                        records[number] = record
            evidence[ident] = {
                "records": [records[n] for n in sorted(records)],
                "status": "ok"
                if records
                else (statuses[-1] if statuses else "no_evidence"),
            }
    cases = []
    for definition in run["scenarios"]:
        if definition["id"] not in evidence:
            raise ValueError("Missing saved evidence for " + definition["id"])
        cases.append({**definition, **evidence[definition["id"]]})
    return cases


class FrozenRAG:
    config = SimpleNamespace(request_timeout=30)

    def __init__(self, backend=None):
        self.case = None
        self.queries = []
        self.backend = backend

    def retrieve(self, query, **kwargs):
        self.queries.append(
            {"query": query, "english_query": kwargs.get("translated_question")}
        )
        if self.backend is not None:
            return self.backend.retrieve(query, **kwargs)
        return {
            "status": self.case["status"],
            "question": query,
            "sources": [record["source"] for record in self.case["records"]],
            "contexts": [record["text"] for record in self.case["records"]],
        }


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, limit: float, spent: float = 0):
        self.limit, self.spent, self.reserved = limit, spent, 0.0

    def reserve(self, amount: float):
        if self.spent + self.reserved + amount > self.limit:
            raise BudgetExceeded("Configured benchmark budget exhausted")
        self.reserved += amount

    def settle(self, reserved: float, actual: float):
        self.reserved -= reserved
        self.spent += actual


class MeasuredModel(AgentModel):
    def __init__(self, settings, budget: Budget, save, calls: list):
        super().__init__(settings, TOOLS)
        self.budget, self.save, self.calls = budget, save, calls
        self.case_id, self.mode = "", "brief"

    async def ainvoke(self, messages, *, max_tokens=500, expanded=False, **kwargs):
        # Reserve the entire configured context, including encrypted reasoning/tools.
        max_input = 128000
        cap = self.output_budget(max_tokens, expanded=expanded)
        prices = RATES[self.model]
        reservation = (max_input * max(prices[:3]) + cap * prices[3]) / 1e6
        self.budget.reserve(reservation)
        row = {
            "case_id": self.case_id,
            "mode": "expanded" if expanded else "brief",
            "status": "started",
        }
        self.calls.append(row)
        self.save()
        started = time.perf_counter()
        try:
            response = await super().ainvoke(
                messages, max_tokens=max_tokens, expanded=expanded, **kwargs
            )
            usage = dict(response.usage_metadata or {})
            cost = usage_cost(self.model, usage)
            self.budget.settle(reservation, cost)
            row.update(
                status="completed",
                usage=usage,
                cost_usd=cost,
                finish_reason=response.response_metadata.get("finish_reason"),
                provider_status=response.response_metadata.get("status"),
                tool_calls=[c["name"] for c in response.tool_calls],
            )
            return response
        except Exception as exc:
            # Unknown usage is conservatively charged at reservation, not zero.
            if row["status"] == "started":
                self.budget.settle(reservation, reservation)
            row.update(
                status="failed",
                reserved_cost_usd=reservation,
                error={
                    "type": type(exc).__name__,
                    "http_status": getattr(exc, "status_code", None),
                    "cause_type": type(exc.__cause__).__name__
                    if exc.__cause__
                    else None,
                    "frames": [
                        f"{Path(frame.f_code.co_filename).name}:{line}:{frame.f_code.co_name}"
                        for frame, line in traceback.walk_tb(exc.__traceback__)
                    ],
                },
            )
            raise
        finally:
            row["latency_ms"] = (time.perf_counter() - started) * 1000
            self.save()


def aggregate(rows: list[dict], calls: list[dict]) -> dict:
    completed = [r for r in rows if r["status"] == "completed"]
    latencies = sorted(r["latency_ms"] for r in completed)
    usages = [c["usage"] for c in calls if c.get("usage")]
    return {
        "planned_attempts": len(rows),
        "completed": len(completed),
        "failed": sum(r["status"] == "failed" for r in rows),
        "incomplete": sum(
            r.get("answer_info", {}).get("incomplete", False) for r in completed
        ),
        "median_ms": statistics.median(latencies) if latencies else None,
        "p95_ms": latencies[math.ceil(len(latencies) * 0.95) - 1]
        if latencies
        else None,
        "generation_cost_usd": sum(c.get("cost_usd", 0) for c in calls),
        "unknown_cost_reservations_usd": sum(
            c.get("reserved_cost_usd", 0) for c in calls
        ),
        "input_tokens": sum(u.get("input_tokens", 0) for u in usages),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usages),
        "reasoning_tokens": sum(
            (u.get("output_token_details") or {}).get("reasoning", 0) or 0
            for u in usages
        ),
        "cache_read_tokens": sum(
            (u.get("input_token_details") or {}).get("cache_read", 0) or 0
            for u in usages
        ),
        "cache_write_tokens": sum(
            (u.get("input_token_details") or {}).get("cache_creation", 0) or 0
            for u in usages
        ),
    }


async def run(args):
    baseline = args.baseline
    raw_run, raw_cp = (
        (baseline / "run.json").read_bytes(),
        (baseline / "checkpoints.json").read_bytes(),
    )
    source, cp = json.loads(raw_run), json.loads(raw_cp)
    if cp["run_sha256"] != hashlib.sha256(raw_run).hexdigest():
        raise ValueError("Checkpoint export does not match baseline")
    cases = extract_cases(source, cp)
    if args.case_ids:
        requested = set(args.case_ids.split(","))
        cases = [c for c in cases if c["id"] in requested]
    if args.limit:
        cases = cases[: args.limit]
    names = args.variants.split(",")
    if any(n not in VARIANTS for n in names):
        raise ValueError("Unknown variant")
    path = args.output
    signature = {
        "baseline_run_sha256": hashlib.sha256(raw_run).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(raw_cp).hexdigest(),
        "case_ids": [c["id"] for c in cases],
        "variants": names,
        "prompt_sha256": hashlib.sha256(PERSISTENT_SYSTEM_PROMPT.encode()).hexdigest(),
        "reasoning_reserve": args.reasoning_reserve,
        "retrieval": args.retrieval,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                ROOT / "app/services/runtime_model.py",
                ROOT / "app/services/agent_persistent.py",
                ROOT / "app/chat/agent.py",
            )
        },
    }
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["signature"] != signature:
            raise ValueError("Resume configuration differs; use a new output path")
    else:
        payload = {
            "schema_version": 1,
            "pricing_verified": "2026-10-15",
            "budget_usd": args.budget_usd,
            "signature": signature,
            "cases": cases,
            "variants": {},
            "method": f"persistent_graph_{args.retrieval}_retrieval; references scoring-only; no Telegram sends",
            "limitations": [
                "Fixed evidence does not measure retrieval query quality; live retrieval does. Neither tests HTTP/Telegram transport.",
                "Latency is graph completion, not streaming TTFT.",
                "Cost covers generation. Live query embeddings/condense are excluded; fixed retrieval has no new embedding cost.",
                "Interrupted/failed conversations are never automatically replayed.",
            ],
        }
    budget = Budget(
        args.budget_usd,
        sum(v.get("spent_usd", 0) for v in payload["variants"].values()),
    )

    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        for variant in payload["variants"].values():
            variant["summary"] = aggregate(variant["results"], variant["calls"])
            variant["spent_usd"] = (
                variant["summary"]["generation_cost_usd"]
                + variant["summary"]["unknown_cost_reservations_usd"]
            )
        atomic_json(path, payload)

    settings = get_settings()
    live_rag = None
    if args.retrieval == "live":
        from dataclasses import replace

        from app.services.rag import RAGConfig, RAGService

        live_config = replace(
            RAGConfig.from_settings(settings),
            ingest_on_start=False,
            rebuild_on_start=False,
        )
        live_rag = RAGService(live_config).build()
    gate = asyncio.Semaphore(args.concurrency)
    models = []

    async def variant_run(name):
        async with gate:
            model_id, effort, api = VARIANTS[name]
            variant = payload["variants"].setdefault(
                name,
                {
                    "model": model_id,
                    "effort": effort,
                    "api": api,
                    "results": [],
                    "calls": [],
                },
            )
            updates = {
                "default_model": model_id,
                "agent_reasoning_effort": effort,
                "agent_expanded_reasoning_effort": effort,
                "agent_llm_api": api,
                "agent_reasoning_token_reserve": args.reasoning_reserve,
            }
            model = MeasuredModel(
                settings.model_copy(update=updates), budget, save, variant["calls"]
            )
            models.append(model)
            rag, saver = FrozenRAG(live_rag), InMemorySaver()
            graph = build_agent(saver, model=model, rag_provider=lambda: rag)
            stop_variant = False
            for case in cases:
                if stop_variant:
                    break
                existing = [
                    r
                    for r in variant["results"]
                    if r["conversation"] == case["conversation"]
                ]
                # In-memory conversation cannot be reconstructed without re-billing.
                if existing:
                    if any(r["case_id"] == case["id"] for r in existing):
                        continue
                    if not any(r.get("_current_process") for r in existing):
                        raise ValueError(
                            "Cannot resume an unfinished dependent conversation; use a new output"
                        )
                rag.case, rag.queries = case, []
                config = {
                    "configurable": {
                        "thread_id": f"luna-{name}-{case['conversation']}",
                        "user_role": "read-only",
                        "response_tokens": 500,
                        "context_budget": 110000,
                    }
                }
                model.case_id = case["id"]
                for expanded in (
                    (False, True)
                    if case.get("golden_index") in (1, 10, 16, 24, 32)
                    else (False,)
                ):
                    ident = case["id"] + (":expanded" if expanded else "")
                    if any(r["case_id"] == ident for r in variant["results"]):
                        continue
                    config["configurable"].update(
                        expanded_mode=expanded,
                        response_tokens=1500 if expanded else 500,
                    )
                    row = {
                        "case_id": ident,
                        "conversation": case["conversation"],
                        "mode": "expanded" if expanded else "brief",
                        "status": "started",
                    }
                    variant["results"].append(row)
                    save()
                    started = time.perf_counter()
                    try:
                        state = await asyncio.wait_for(
                            graph.ainvoke(
                                {
                                    "messages": [
                                        HumanMessage(
                                            content="Раскрой предыдущий ответ подробнее."
                                            if expanded
                                            else case["question"]
                                        )
                                    ]
                                },
                                config,
                            ),
                            timeout=240,
                        )
                        canonical = ChatAgent._result(state, None)
                        row.update(
                            status="completed",
                            answer=canonical["answer"],
                            sources=canonical["sources"],
                            answer_info=canonical["answer_info"],
                            queries=list(rag.queries),
                            evidence=state.get("citation_evidence", []),
                            _current_process=True,
                        )
                    except Exception as exc:  # noqa: BLE001 - never store provider bodies
                        row.update(
                            status="failed",
                            error={
                                "type": type(exc).__name__,
                                "http_status": getattr(exc, "status_code", None),
                            },
                        )
                    row["latency_ms"] = (time.perf_counter() - started) * 1000
                    save()
                    print(
                        json.dumps(
                            {
                                "variant": name,
                                "case": ident,
                                "status": row["status"],
                                "spent_usd": round(budget.spent, 4),
                            }
                        ),
                        flush=True,
                    )
                    if row["status"] != "completed":
                        stop_variant = True
                        break

    await asyncio.gather(*(variant_run(n) for n in names))
    # LangChain caches HTTP transports across model instances. Closing a finished
    # variant early would break other in-flight variants sharing that transport.
    for model in models:
        base = getattr(model.client, "bound", model.client)
        await base.root_async_client.close()
        base.root_client.close()
    if live_rag is not None:
        live_rag.close()
    for variant in payload["variants"].values():
        for row in variant["results"]:
            row.pop("_current_process", None)
    save()
    print(
        json.dumps({name: v["summary"] for name, v in payload["variants"].items()}),
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--budget-usd", type=float, default=5)
    parser.add_argument("--reasoning-reserve", type=int, default=8192)
    parser.add_argument("--retrieval", choices=["fixed", "live"], default="fixed")
    args = parser.parse_args()
    if args.concurrency < 1 or args.budget_usd <= 0 or args.reasoning_reserve < 0:
        parser.error("Invalid concurrency, budget or reserve")
    return args


if __name__ == "__main__":
    try:
        asyncio.run(run(parse_args()))
    except Exception as exc:  # noqa: BLE001 - safe CLI diagnostics
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1) from None
