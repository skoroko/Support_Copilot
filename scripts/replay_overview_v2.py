"""No-network v1/v2 replay on the identical four captured retrieval pools."""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
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

from app.chat.service import count_tokens
from app.services import overview_evidence
from app.services.agent_persistent import build_agent
from scripts.compare_overview_prefetch import frozen_inputs


def load_saved(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "var/overview_prefetch_ab/evaluated_code" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


async def replay():
    old_overview = load_saved("overview_v1_snapshot", "overview_evidence.py")
    try:
        sys.modules["app.services.overview_evidence"] = old_overview
        old_graph = load_saved("agent_v1_snapshot", "agent_persistent.py").build_agent
    finally:
        sys.modules["app.services.overview_evidence"] = overview_evidence
    rows = []
    for slot in (1, 2, 4, 7):
        sample = json.loads((ROOT / f"var/overview_prefetch_ab/sample_{slot:02}.json").read_text(encoding="utf-8"))
        case = next(c for c in frozen_inputs() if c["id"] == sample["question"])
        stored = {r["args"][0]: r["result"] for r in sample["retrievals"][:2]}
        snapshots = []
        for version, builder in (("v1", old_graph), ("v2", build_agent)):
            class StoredRAG:
                supports_overview_prefetch = True
                config = SimpleNamespace(request_timeout=5, rerank_enabled=False)

                def __init__(self, data):
                    self.data, self.calls = data, 0

                def retrieve(self, query, **kwargs):
                    self.calls += 1
                    return deepcopy(self.data[query])

            class CapturedModel:
                def __init__(self):
                    self.calls, self.tokens = 0, 0

                async def ainvoke(self, messages, **kwargs):
                    self.calls += 1
                    self.tokens = count_tokens([{"role": m.type, "content": str(m.content)} for m in messages])
                    return AIMessage(content="Офлайн проверка источников. [1]")

            rag, model = StoredRAG(stored), CapturedModel()
            graph = builder(InMemorySaver(), model=model, rag_provider=lambda rag=rag: rag)
            state = await graph.ainvoke({"messages": messages_from_dict(case["messages"])}, {"configurable": {
                "thread_id": f"v2-replay-{slot}-{version}", "overview_prefetch_enabled": True,
                "history_window": sample["settings"]["history_window"],
                "context_budget": sample["settings"]["context_budget"]}})
            records = state["citation_evidence"]
            snapshots.append(records)
            assert model.calls == 1 and rag.calls == (2 if version == "v1" else 1)
            rows.append({"slot": slot, "question": case["id"], "version": version,
                         "stored_retrieval_calls": rag.calls, "scripted_model_calls": model.calls,
                         "local_prompt_token_estimate": model.tokens, "sources": len(records),
                         "context_chars": sum(len(e["text"]) for e in records)})
        for record in snapshots[1]:
            old = next(r for r in snapshots[0] if r["source"]["id"] == record["source"]["id"])
            assert record["text"] in old["text"]
            for key in ("file_name", "page", "printed_page"):
                assert record["source"].get(key) == old["source"].get(key)
    return {"network_calls": 0, "rows": rows, "v2_evidence_is_unmodified_subset_of_v1": True,
            "limitations": "Stored retrievals and cooperative fake model; no live model latency or quality claim."}


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = asyncio.run(replay())
    target = ROOT / "tests/eval/results/2026-10-16_overview_prefetch/v2_offline.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
