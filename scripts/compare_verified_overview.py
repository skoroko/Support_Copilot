"""Serial live A/B for the narrow HSS pilot; full answers remain in ignored var/.

Run in an isolated candidate-image Compose container. Never sends Telegram,
ingests documents, or touches production conversations/checkpoints.
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
# The first completed series is immutable; a changed candidate uses a new run.
RUN_ID = sys.argv[sys.argv.index("--run-id") + 1] if "--run-id" in sys.argv else "verified_overview_live"
assert RUN_ID in {"verified_overview_live", "verified_overview_live_v2"}
OUT = ROOT / "var" / RUN_ID
CASES = [
    {"id": "hss_explicit", "messages": [
        ("human", "Расскажи про модель Hardening Soil small strain в GTS NX.")]},
    {"id": "hss_title_alias", "messages": [
        ("human", "Что такое Hardening Soil (small strain stiffness) в GTS NX?")]},
    {"id": "hss_after_hs", "messages": [
        ("human", "В этой беседе ранее обсуждалась базовая модель Hardening Soil."),
        ("ai", "Следующий самостоятельный вопрос рассмотрим отдельно, по документации."),
        ("human", "Дай краткий обзор модели Hardening Soil small strain в GTS NX.")]},
]
SCHEDULE = [
    {"slot": 6 * repeat + 2 * case_index + offset, "question": case["id"],
     "repeat": repeat, "arm": arm}
    for repeat in range(3) for case_index, case in enumerate(CASES)
    for offset, arm in enumerate(("ordinary", "pilot") if (repeat + case_index) % 2 == 0
                                 else ("pilot", "ordinary"))
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def manifest() -> dict:
    # Match the Dockerfile/.dockerignore runtime boundary, excluding CLI logs.
    code = sorted(p for folder in ("app", "bot", "support_copilot/tools",
                                   "support_copilot/prompts", "support_copilot/data")
                  for p in (ROOT / folder).rglob("*")
                  if p.is_file() and p.suffix in {".py", ".txt", ".json"}
                  and "__pycache__" not in p.parts)
    code.extend([ROOT / "support_copilot/__init__.py", ROOT / "support_copilot/config.py",
                 Path(__file__)])
    return {
        "protocol": "verified_hss_live_v1", "run_id": RUN_ID, "schedule": SCHEDULE,
        "cases": json.loads(json.dumps(CASES)),
        "comparison": "same candidate code and runtime profile; flag false vs true",
        "concurrency": 1, "fresh_process_per_sample": True,
        "local_tokenizer_warmed_before_timing": True,
        "rag_initialization_excluded_from_answer_time": True,
        "service_tier": "default", "external_destination": "OpenAI API",
        "telegram_allowed": False, "production_mutation_allowed": False,
        "generation_answer_limit": 18, "attempts_per_slot": 1,
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in code},
        "release_gate": {
            "latency": "pilot median <= 0.8 * ordinary median; pilot faster in >= 7/9 pairs",
            "all_pilot_answers": "verified route, complete useful answer, one generation call",
            "accuracy": "zero material unsupported or incorrect technical claims in pilot",
            "completeness": "all four required facets in every pilot answer; no worse paired score",
            "citations": "all technical claims supported by their exact cited text; no wrong variant",
            "facets": ["explicit HSS and Modified Mohr Coulomb basis",
                       "small-strain stiffness behavior",
                       "E50ref/Eoedref/Eurref and distinct meanings",
                       "G0ref and gamma0.7 with correct meaning"],
            "failures": "all failures retained, no successful-answer replacement",
            "scope": "brief explicit HSS overview only; not general quality or latency SLA",
        },
    }


async def sample(slot: int) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict
    from langgraph.checkpoint.memory import InMemorySaver

    from app.chat.agent import ChatAgent
    from app.chat.service import _o200k_base_encoding
    from app.core.config import get_settings
    from app.services.agent_persistent import (
        TOOLS,
        build_agent,
        retrieval_worker_tracker,
    )
    from app.services.rag import RAGConfig, RAGService
    from app.services.runtime_model import AgentModel

    assert 0 <= slot < len(SCHEDULE)
    assert manifest() == json.loads((OUT / "manifest.json").read_text(encoding="utf-8")), "Code drift"
    target = OUT / f"sample_{slot:02}.json"
    assert not target.exists(), "Slot already attempted"
    spec = SCHEDULE[slot]
    settings = get_settings()
    assert settings.default_model == "gpt-6-luna"
    assert settings.agent_reasoning_effort == "medium"
    assert settings.agent_llm_api == "responses"
    pilot = spec["arm"] == "pilot"
    calls, retrievals, preparation = [], [], []
    row = dict(spec, status="started", started_epoch=time.time(), model_calls=calls,
               retrievals=retrievals, selection_seconds=preparation)
    save(target, row)
    rag = None
    try:
        started = time.perf_counter()
        encoding = _o200k_base_encoding()
        assert encoding is not None, "Tokenizer unavailable"
        encoding.encode("Одинаковый прогрев tokenizer перед измерением")
        row["tokenizer_initialization_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        rag = RAGService(replace(RAGConfig.from_settings(settings), ingest_on_start=False,
                                 evidence_scope_enabled=pilot)).build()
        row["rag_initialization_seconds"] = time.perf_counter() - started

        class TimedModel(AgentModel):
            async def ainvoke(self, *args, **kwargs):
                call = {"status": "started", "started_epoch": time.time()}
                calls.append(call)
                save(target, row)
                started = time.perf_counter()
                try:
                    result = await super().ainvoke(*args, service_tier="default", **kwargs)
                    call.update(status="completed", seconds=time.perf_counter() - started,
                                service_tier=result.response_metadata.get("service_tier"),
                                usage=result.usage_metadata, tool_calls=len(result.tool_calls),
                                finish_reason=result.response_metadata.get("finish_reason"))
                    assert call["service_tier"] == "default", "Service tier mismatch"
                    return result
                except Exception as exc:
                    call.update(status="failed", seconds=time.perf_counter() - started,
                                error_type=type(exc).__name__, failed_call_usage="unknown")
                    raise
                finally:
                    save(target, row)

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
                save(target, row)

        rag.retrieve = timed_retrieve

        async def forbidden_send(*args, **kwargs):
            raise AssertionError("Telegram forbidden in evaluation")

        model = TimedModel(settings, TOOLS)
        context_budget = (settings.chat_model_context_tokens - settings.chat_response_tokens
                          - settings.agent_reasoning_token_reserve - settings.chat_safety_margin_tokens)
        row["settings"] = {"model": model.model, "api": settings.agent_llm_api,
                           "effort": model.effort, "answer_tokens": settings.chat_response_tokens,
                           "generation_cap": model.output_budget(settings.chat_response_tokens),
                           "history_window": settings.chat_context_window, "context_budget": context_budget,
                           "evidence_scope_enabled": pilot}
        graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag,
                            send_message=forbidden_send)
        case = next(c for c in CASES if c["id"] == spec["question"])
        messages = [(HumanMessage if role == "human" else AIMessage)(content=content)
                    for role, content in case["messages"]]
        config = {"configurable": {"thread_id": f"verified-hss-live-{slot}",
                  "verified_overview_enabled": pilot, "overview_prefetch_enabled": False,
                  "user_role": "read-only", "response_tokens": settings.chat_response_tokens,
                  "history_window": settings.chat_context_window, "context_budget": context_budget}}
        workers = []
        token = retrieval_worker_tracker.set(workers.append)
        started = time.perf_counter()
        try:
            state = await graph.ainvoke({"messages": messages}, config)
            elapsed = time.perf_counter() - started
        finally:
            retrieval_worker_tracker.reset(token)
            await asyncio.gather(*workers, return_exceptions=True)
        row.update(status="completed", total_seconds=elapsed,
                   answer=ChatAgent._result(state, None), citation_evidence=state.get("citation_evidence"),
                   route=state.get("overview_route"), verified=state.get("verified_overview"),
                   packet=state.get("overview_packet"), messages=messages_to_dict(state["messages"]))
    except Exception as exc:  # noqa: BLE001 - private boundary, provider errors may contain credentials
        row.update(status="failed", error_type=type(exc).__name__)
    finally:
        save(target, row)
        if rag is not None:
            rag.client.close()
    return {k: row.get(k) for k in ("slot", "question", "repeat", "arm", "status", "route", "error_type")} | {
        "seconds": round(row.get("total_seconds", 0), 3), "model_calls": len(calls),
        "retrieval_calls": len(retrievals)}


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    if sys.argv[1] == "manifest":
        OUT.mkdir(parents=True, exist_ok=True)
        assert not (OUT / "manifest.json").exists()
        save(OUT / "manifest.json", manifest())
        print(json.dumps({"protocol": "verified_hss_live_v1", "samples": len(SCHEDULE)}))
    else:
        assert "--allow-paid-api" in sys.argv, "Explicit paid API acknowledgement required"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = asyncio.run(sample(int(sys.argv[1])))
        print(json.dumps(result))
        raise SystemExit(0 if result["status"] == "completed" else 1)
