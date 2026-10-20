"""Concurrent read-only work must retain evidence and ordered citations."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.services.agent_persistent import build_agent
from app.services.parallel_retrieval import bilingual_searches, recall_slots
from app.services.rag import RAGConfig, RAGService
from app.services.turn_budget import remaining_seconds, turn_budget


def test_english_recall_starts_before_slow_original_finishes():
    original_started, english_started = Event(), Event()

    def search(query):
        if query == "original":
            original_started.set()
            assert english_started.wait(2)
        else:
            english_started.set()
        return [query + " exact evidence"]

    def translate():
        assert original_started.wait(2)
        return "english"

    rows, translated, omitted = bilingual_searches(["original"], search, translate)
    assert translated == "english" and not omitted
    assert [row.items for row in rows] == [["original exact evidence"], ["english exact evidence"]]
    assert all(row.error_type is None for row in rows)


def test_all_language_variants_share_one_search_concurrency_cap():
    lock, barrier = Lock(), Barrier(2, timeout=2)
    active = peak = 0

    def search(query):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait()
        with lock:
            active -= 1
        return [query]

    rows, _, _ = bilingual_searches(["a", "b", "c"], search, lambda: "english", concurrency=2)
    assert peak == 2
    assert [row.items for row in rows] == [["a"], ["b"], ["c"], ["english"]]


@pytest.mark.parametrize("translation,limit,expected,omitted", [
    ("english", 1, ["original"], True),
    ("original", 1, ["original"], False),
    (None, None, ["original"], False),
])
def test_duplicate_and_budget_rules_preserve_original_recall(translation, limit, expected, omitted):
    rows, _, actual_omitted = bilingual_searches(["original"], lambda q: [q],
        lambda: translation, query_limit=limit)
    assert [row.query for row in rows] == expected
    assert actual_omitted is omitted


def test_translation_and_sibling_failure_do_not_erase_successful_evidence():
    def search(query):
        if query == "failed":
            raise RuntimeError("private body")
        return ["supported fact"]

    rows, _, _ = bilingual_searches(["failed", "works"], search, lambda: "english")
    assert rows[0].error_type == "RuntimeError"
    assert rows[1].items == rows[2].items == ["supported fact"]
    assert "private body" not in repr(rows)


def test_parallel_lexical_queries_never_mutate_shared_index_or_sibling_metadata():
    from llama_index.core.schema import TextNode

    original = TextNode(text="exact supported fact", metadata={"nested": {"queries": []}})
    service = RAGService(RAGConfig.from_settings(SimpleNamespace()))
    service.client = object()
    service._lexical_bm25 = SimpleNamespace(query=lambda *_: [(original, 1.0)])
    barrier = Barrier(2, timeout=2)

    def query(label):
        item = service._retrieve_lexical_candidates(label)[0]
        item.node.metadata["_lexical_query_hashes"] = [label]
        item.node.metadata["nested"]["queries"].append(label)
        barrier.wait()
        return item.node

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(query, ["first", "second"]))
    assert first.metadata["_lexical_query_hashes"] == ["first"]
    assert second.metadata["_lexical_query_hashes"] == ["second"]
    assert first.metadata["nested"]["queries"] == ["first"]
    assert second.metadata["nested"]["queries"] == ["second"]
    assert original.metadata == {"nested": {"queries": []}}


@pytest.mark.asyncio
async def test_english_and_original_workers_inherit_the_same_turn_deadline():
    async with turn_budget(2):
        rows, _, _ = await asyncio.to_thread(bilingual_searches, ["original"],
            lambda _: [remaining_seconds(10)], lambda: "english")
        assert all(0 < row.items[0] <= 2 for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_independent_tools_run_together_and_merge_citations_in_call_order(monkeypatch, failed):
    monkeypatch.setattr("app.chat.service.count_tokens", lambda messages: len(str(messages)) // 4)
    barrier, second_done = Barrier(2, timeout=2), Event()
    completed = []
    seen_slots = []

    def retrieve(query, **kwargs):
        seen_slots.append(recall_slots.get())
        assert kwargs["translated_question"] == query
        barrier.wait()
        if query == "first":
            assert second_done.wait(2)
            if failed:
                raise TimeoutError("private provider body")
        else:
            second_done.set()
        completed.append(query)
        return {"status": "ok", "sources": [{"chunk_id": query,
            "file_name": "manual.pdf", "product": "gts_nx", "page": 1}],
            "contexts": [query + " supported fact."], "telemetry": {}}

    class Model:
        step = 0

        async def ainvoke(self, messages, **kwargs):
            self.step += 1
            if self.step == 1:
                return AIMessage(content="", tool_calls=[{"name": "search_knowledge_base",
                    "id": query, "args": {"query": query, "english_query": query}}
                    for query in ["first", "second"]])
            tools = [json.loads(m.content) for m in messages if isinstance(m, ToolMessage)]
            assert len(tools) == 2
            if failed:
                assert tools[0]["status"] == "unavailable"
                assert tools[1]["citation_numbers"] == [1]
                return AIMessage(content="second supported fact [1].")
            assert [tool["citation_numbers"] for tool in tools] == [[1], [2]]
            return AIMessage(content="first supported fact [1]. second supported fact [2].")

    rag = SimpleNamespace(retrieve=retrieve, config=SimpleNamespace(request_timeout=3))
    graph = build_agent(InMemorySaver(), model=Model(), rag_provider=lambda: rag)
    state = await graph.ainvoke({"messages": [HumanMessage(content="Which facts are supported?")]},
        {"configurable": {"thread_id": "ordered-batch", "context_budget": 20000}})
    assert completed == (["second"] if failed else ["second", "first"])
    assert [r["source"]["chunk_id"] for r in state["citation_evidence"]] == (
        ["second"] if failed else ["first", "second"])
    assert "private provider body" not in str(state["messages"])
    assert seen_slots[0] is seen_slots[1] and seen_slots[0] is not None
    assert recall_slots.get() is None
