"""Isolated serial model A/B; private artifacts in var, stdout metadata only.

Run inside the existing app container, one fresh process per numbered slot.
Never fetch conversation data, alter settings, or send Telegram messages.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "var" / "optimized_model_ab"
INPUT = ROOT / "var" / "latency_inputs_real.json"
CODE = ("app/services/agent_persistent.py", "app/services/evidence_text.py",
        "app/services/rag.py", "app/services/runtime_model.py")
PROFILES = {
    "mini": {"default_model": "gpt-5.4-mini", "agent_llm_api": "chat_completions",
             "agent_reasoning_effort": "none", "agent_expanded_reasoning_effort": "none"},
    "luna": {"default_model": "gpt-6-luna", "agent_llm_api": "responses",
             "agent_reasoning_effort": "medium", "agent_expanded_reasoning_effort": "medium"},
}
# Alternating first/second position per question; 3 replicates, concurrency 1.
SCHEDULE = [(q, r, m) for r in range(3) for q in ("0801f604b07c", "a7b5f7c44544")
            for m in (("mini", "luna") if (r + (q == "a7b5f7c44544")) % 2 == 0
                      else ("luna", "mini"))]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def frozen_inputs() -> list[dict]:
    cases = json.loads(INPUT.read_text(encoding="utf-8"))
    assert {c["id"] for c in cases} == {q for q, _, _ in SCHEDULE}
    for case in cases:
        assert sha(case["messages"][-1]["data"]["content"].encode())[:12] == case["id"]
        for message in case["messages"]:
            # Preserve all text and tool history; discard cross-model transport metadata.
            message["data"]["additional_kwargs"] = {}
            message["data"]["response_metadata"] = {}
            message["data"]["usage_metadata"] = None
    return cases


def manifest() -> dict:
    cases = frozen_inputs()
    return {"protocol": 1, "concurrency": 1, "fresh_process_per_sample": True,
            "profiles": PROFILES, "history_metadata_removed_for_both": True,
            "input_file_sha256": sha(INPUT.read_bytes()),
            "semantic_inputs_sha256": sha(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()),
            "code_sha256": {p: sha((ROOT / p).read_bytes()) for p in CODE},
            "schedule": [{"slot": i, "question": q, "repeat": r, "profile": m}
                         for i, (q, r, m) in enumerate(SCHEDULE)]}


async def sample(slot: int, attempt: int) -> dict:
    from langchain_core.messages import messages_from_dict, messages_to_dict
    from langgraph.checkpoint.memory import InMemorySaver

    from app.chat.agent import ChatAgent
    from app.core.config import get_settings
    from app.services.agent_persistent import (
        PERSISTENT_SYSTEM_PROMPT,
        TOOLS,
        build_agent,
    )
    from app.services.rag import RAGConfig, RAGService
    from app.services.runtime_model import AgentModel

    assert 0 <= slot < 12 and attempt in (1, 2)
    OUT.mkdir(parents=True, exist_ok=True)
    expected = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest() == expected, "Code or inputs changed since manifest"
    assert not (OUT / f"sample_{slot:02}.json").exists(), "Successful slot already exists"
    target = OUT / f"attempt_{slot:02}_{attempt}.json"
    assert not target.exists(), "Attempt already used"
    if attempt == 2:
        failure = json.loads((OUT / f"attempt_{slot:02}_1.json").read_text())
        assert failure["status"] == "failed" and failure["error_type"] in (
            "APIConnectionError", "OpenAIConnectionError", "APITimeoutError")
    q, repeat, profile = SCHEDULE[slot]
    production = get_settings()
    assert production.default_model == "gpt-6-luna" and production.agent_reasoning_effort == "medium"
    settings = production.model_copy(update=PROFILES[profile])
    calls, retrievals, preparation = [], [], []
    row = {"slot": slot, "attempt": attempt, "question": q, "repeat": repeat,
           "profile": profile, "status": "started", "model_calls": calls,
           "retrievals": retrievals, "selection_seconds": preparation,
           "prompt_sha256": sha(PERSISTENT_SYSTEM_PROMPT.encode())}
    save(target, row)
    rag = None
    try:
        rag = RAGService(replace(RAGConfig.from_settings(production), ingest_on_start=False)).build()
        class TimedModel(AgentModel):
            async def ainvoke(self, *args, **kwargs):
                started = time.perf_counter()
                result = await super().ainvoke(*args, **kwargs)
                calls.append({"seconds": time.perf_counter() - started,
                              "usage": result.usage_metadata,
                              "tool_calls": len(result.tool_calls),
                              "finish_reason": result.response_metadata.get("finish_reason")})
                return result
        select = rag._select_context
        def timed_select(*args, **kwargs):
            started = time.perf_counter()
            try:
                return select(*args, **kwargs)
            finally:
                preparation.append(time.perf_counter() - started)
        rag._select_context = timed_select
        retrieve = rag.retrieve
        def timed_retrieve(*args, **kwargs):
            started = time.perf_counter()
            record = {"args": args, "kwargs": kwargs}
            try:
                result = retrieve(*args, **kwargs)
                record.update(status="completed", result=result)
                return result
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__)
                raise
            finally:
                record["seconds"] = time.perf_counter() - started
                retrievals.append(record)
        rag.retrieve = timed_retrieve
        async def forbidden_send(*args, **kwargs):
            raise AssertionError("Telegram forbidden in evaluation")
        model = TimedModel(settings, TOOLS)
        row["settings"] = {"model": model.model, "api": settings.agent_llm_api,
                           "effort": model.effort, "answer_tokens": settings.chat_response_tokens,
                           "generation_cap": model.output_budget(settings.chat_response_tokens),
                           "history_window": settings.chat_context_window,
                           "context_budget": production.chat_model_context_tokens - production.chat_response_tokens
                           - production.agent_reasoning_token_reserve - production.chat_safety_margin_tokens}
        graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag,
                            send_message=forbidden_send)
        case = next(c for c in frozen_inputs() if c["id"] == q)
        config = {"configurable": {"thread_id": f"optimized-ab-{slot}-{attempt}",
                  "user_role": "read-only", "response_tokens": settings.chat_response_tokens,
                  "history_window": settings.chat_context_window,
                  "context_budget": row["settings"]["context_budget"]}}
        started = time.perf_counter()
        state = await graph.ainvoke({"messages": messages_from_dict(case["messages"])}, config)
        row.update(status="completed", total_seconds=time.perf_counter() - started,
                   answer=ChatAgent._result(state, None), citation_evidence=state.get("citation_evidence"),
                   messages=messages_to_dict(state["messages"]))
        save(OUT / f"sample_{slot:02}.json", row)
    except Exception as exc:  # noqa: BLE001 - private runner boundary; never print provider errors
        row.update(status="failed", error_type=type(exc).__name__, failed_call_usage="unknown")
    finally:
        save(target, row)
        if rag is not None:
            rag.client.close()
    return {k: row[k] for k in ("slot", "attempt", "question", "profile", "status")} | {
        "total": round(row.get("total_seconds", 0), 3), "model_calls": len(calls),
        "model_seconds": round(sum(c["seconds"] for c in calls), 3),
        "retrievals": len(retrievals),
        "retrieval_seconds": round(sum(c["seconds"] for c in retrievals), 3),
        "preparation_seconds": round(sum(preparation), 3), "error_type": row.get("error_type")}


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    if sys.argv[1] == "manifest":
        OUT.mkdir(parents=True, exist_ok=True)
        assert not (OUT / "manifest.json").exists()
        save(OUT / "manifest.json", manifest())
        print(json.dumps(manifest(), indent=2))
    else:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = asyncio.run(sample(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 1))
        print(json.dumps(result))
        raise SystemExit(0 if result["status"] == "completed" else 1)
