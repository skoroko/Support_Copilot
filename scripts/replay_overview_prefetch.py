"""Offline graph replay using private frozen inputs and previously returned evidence.

This tests orchestration/packaging, not the recall of the new search queries or
the choices/quality of a live model. No provider clients are constructed.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, messages_from_dict
from langgraph.checkpoint.memory import InMemorySaver

from app.services.agent_persistent import build_agent
from app.services.overview_evidence import plan_overview


async def replay() -> dict:
    inputs = ROOT / "var/latency_inputs_real.json"
    cases = json.loads(inputs.read_text(encoding="utf-8"))
    rows = []
    for slot in (1, 4, 7, 9, 10):  # All five substantive historical Luna answers.
        sample = json.loads((ROOT / f"var/optimized_model_ab/sample_{slot:02}.json").read_text(encoding="utf-8"))
        case = next(c for c in cases if c["id"] == sample["question"])
        messages = messages_from_dict(deepcopy(case["messages"]))
        plan = plan_overview(messages[-1].content)
        assert plan is not None
        results = [r["result"] for r in sample["retrievals"] if r.get("result", {}).get("status") == "ok"]
        assert results
        facets = {query: results[min(i, len(results) - 1)] for i, (query, _) in enumerate(plan.queries)}

        class StoredRAG:
            supports_overview_prefetch = True
            config = SimpleNamespace(request_timeout=5, rerank_enabled=False)

            def __init__(self, stored_facets, default):
                self.calls = 0
                self.facets = stored_facets
                self.default = default

            def retrieve(self, query, **kwargs):
                self.calls += 1
                return deepcopy(self.facets.get(query, self.default))

        class CooperativeModel:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, prompt, **kwargs):
                self.calls += 1
                supplied = any("Предварительный пакет evidence" in str(m.content) for m in prompt)
                if self.calls == 1 and not supplied:
                    return AIMessage(content="", tool_calls=[{"name": "search_knowledge_base",
                        "args": {"query": "stored recall"}, "id": "offline-search"}])
                return AIMessage(content="Доказательства подготовлены для ответа. [1]")

        for enabled in (False, True):
            rag, model = StoredRAG(facets, results[0]), CooperativeModel()
            graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda rag=rag: rag)
            state = await graph.ainvoke({"messages": deepcopy(messages)}, {"configurable": {
                "thread_id": f"offline-{slot}-{enabled}", "overview_prefetch_enabled": enabled}})
            records = state["citation_evidence"]
            assert state["messages"][:len(messages)] == messages
            assert [r["citation_number"] for r in records] == list(range(1, len(records) + 1))
            assert sum(len(r["text"]) for r in records) <= 24_000
            assert model.calls == (1 if enabled else 2)
            assert rag.calls == (2 if enabled else 1)
            rows.append({"historical_slot": slot, "question": case["id"], "prefetch": enabled,
                         "scripted_model_calls": model.calls, "stored_retrieval_calls": rag.calls,
                         "sources": len(records), "context_chars": sum(len(r["text"]) for r in records),
                         "missing_aspects": (state.get("overview_packet") or {}).get("missing_aspects"),
                         "history_unchanged": True, "citations_contiguous": True})
    return {"offline": True, "network_calls": 0, "rows": rows,
            "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
            "limitation": "Stored old-query results; cooperative scripted model. Not a live recall, latency or answer-quality result."}


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = asyncio.run(replay())
    target = ROOT / "tests/eval/results/2026-10-16_overview_prefetch/offline.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
