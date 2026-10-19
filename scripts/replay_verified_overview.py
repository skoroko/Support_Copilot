"""No-network scope/graph replay against frozen chunks and original CHM bytes.

The explicit HSS question is derived for this contract test. Captured retrieval
pools came from older queries, so this measures neither new-query recall nor
live model latency/quality. Private inputs are never included in the report.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import logging
import sys
import tempfile
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.services.agent_persistent import build_agent
from app.services.evidence_scope import evidence_scope
from app.services.evidence_text import evidence_session

QUESTION = "Расскажи про модель Hardening Soil small strain в GTS NX"


async def replay() -> dict:
    original_path = ROOT / "var/latency_after_private.json"
    originals = json.loads(original_path.read_text(encoding="utf-8"))
    nodes = {}
    for row in originals:
        for pool in row["selection_inputs"]:
            batches = [*pool["rankings"], pool.get("lexical_candidates", []), pool.get("translated_candidates", [])]
            for batch in batches:
                for item in batch:
                    node = item["node"]
                    nodes[node["id_"]] = node
    rows, manifests = [], []
    for slot in (1, 4):  # Both captured V1 HSS-bearing prefetch result pools.
        path = ROOT / f"var/overview_prefetch_ab/sample_{slot:02}.json"
        sample = json.loads(path.read_text(encoding="utf-8"))
        manifests.append({"slot": slot, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        results = [deepcopy(item["result"]) for item in sample["retrievals"][:2]]
        scope_counts: Counter = Counter()
        scope_started = time.perf_counter()
        with evidence_session(ROOT / "data"):
            for result in results:
                for source, context in zip(result.get("sources", []), result.get("contexts", []), strict=True):
                    identity = source["chunk_id"]
                    if identity in nodes:
                        node = nodes[identity]
                        source["evidence_scope"] = evidence_scope(
                            ROOT / "data", node["metadata"], identity, node["text"], projected_text=context)
                    else:
                        source["evidence_scope"] = {"status": "unknown", "reason": "original_chunk_not_captured"}
                    scope = source["evidence_scope"]
                    scope_counts[scope.get("subject", scope.get("reason", "unknown"))] += 1
        scope_seconds = time.perf_counter() - scope_started
        for enabled in (False, True):
            class StoredRAG:
                supports_overview_prefetch = True
                config = SimpleNamespace(request_timeout=5, rerank_enabled=False)

                def __init__(self, captured):
                    self.calls = 0
                    self.results = captured

                def retrieve(self, *args, **kwargs):
                    self.calls += 1
                    return deepcopy(self.results[min(self.calls - 1, len(self.results) - 1)])

            class ScriptedModel:
                def __init__(self):
                    self.calls = 0

                async def ainvoke(self, messages, **kwargs):
                    self.calls += 1
                    if self.calls == 1 and kwargs.get("tool_choice") != "none":
                        return AIMessage(content="", tool_calls=[{"name": "search_knowledge_base",
                            "args": {"query": "stored recall"}, "id": "offline-recall"}])
                    return AIMessage(content="Офлайн проверка orchestration, без оценки качества ответа. [1]")

            rag, model = StoredRAG(results), ScriptedModel()
            graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda rag=rag: rag)
            started = time.perf_counter()
            state = await graph.ainvoke({"messages": [HumanMessage(content=QUESTION)]}, {"configurable": {
                "thread_id": f"verified-replay-{slot}-{enabled}", "verified_overview_enabled": enabled}})
            elapsed = time.perf_counter() - started
            assert model.calls == (1 if enabled else 2)
            assert bool(state["verified_overview"]) == enabled
            records = state["citation_evidence"]
            assert [r["marker"] for r in records] == [f"[{i}]" for i in range(1, len(records) + 1)]
            assert state["messages"][0].content == QUESTION
            original_texts = [text for result in results for text in result.get("contexts", [])]
            assert all(r["text"] in original_texts for r in records)
            if enabled:
                assert all(r["source"]["evidence_scope"]["subject"] == "Hardening Soil small strain" for r in records)
            rows.append({"slot": slot, "verified_enabled": enabled, "scripted_model_calls": model.calls,
                         "stored_retrieval_calls": rag.calls, "source_count": len(records),
                         "context_chars": sum(len(r["text"]) for r in records),
                         "source_scope_counts": dict(scope_counts), "scope_seconds": round(scope_seconds, 6),
                         "local_graph_seconds": round(elapsed, 6), "original_texts_unchanged": True,
                         "route": state["overview_route"]})
    return {"network_calls": 0, "paid_model_calls": 0, "rows": rows, "input_manifests": manifests,
            "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                "app/services/agent_persistent.py", "app/services/evidence_scope.py",
                "app/services/evidence_text.py", "app/services/verified_overview.py",
                "app/services/rag.py", "scripts/replay_verified_overview.py")},
            "original_pool_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
            "limitations": "Derived explicit HSS question, frozen older-query retrievals, scripted model. "
                           "No live recall, model latency, answer quality or 15-second SLA claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "var/verified_overview/offline.json")
    args = parser.parse_args()
    # Keep native CHM extraction temporary files inside this workspace on Windows.
    task_temp = ROOT / "var/verified_overview/tmp"
    task_temp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(task_temp)
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        report = asyncio.run(replay())
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
