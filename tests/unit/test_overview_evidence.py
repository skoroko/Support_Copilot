"""Offline contracts for early recall, bounded evidence and normal fallback."""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from threading import Barrier
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.chat.agent import ChatAgent
from app.services.agent_persistent import build_agent
from app.services.overview_evidence import (
    can_skip_second_facet,
    package_overview,
    plan_overview,
)

HS = "расскажи мне про модель харденинг соил"
BOUNDARIES = "какие граничные условия есть в gts nx"
DESCRIPTION = "Hardening Soil small strain stiffness uses Modified Mohr-Coulomb. Stiffness decreases with strain."
PARAMETERS = "Hardening Soil small strain parameters:\nColumn 1=G0ref | Column 2=initial shear modulus\nColumn 1=gamma0.7 | Column 2=70 percent stiffness"
CLASSIFICATION = "DOF constraint for Stress analysis. Seepage boundary for Seepage analysis. Boundary element for Equivalent linear analysis."


def payload(identity, text, *, status="ok"):
    return {"status": status, "sources": [{"id": identity, "chunk_id": identity,
            "file_name": "manual.pdf", "page": 5, "printed_page": 4, "score": .8}],
            "contexts": [text]}


@pytest.mark.parametrize("question", [HS, BOUNDARIES, "расскажи про Soft Soil Creep",
    "Что такое модель Мора-Кулона?", "tell me about Modified Cam Clay in GTS NX",
    "опиши неизвестную модель"])
def test_recognizer_is_explicit(question):
    assert bool(plan_overview(question)) == (question != "опиши неизвестную модель")


@pytest.mark.parametrize("question", ["А её параметры?", "Раскрой предыдущий ответ подробнее.",
    "расскажи про модель Hardening Soil и сравни с Soft Soil", "как настроить Hardening Soil",
    "расскажи про модель Hardening Soil в Civil", "отправь обзор Hardening Soil в Telegram",
    "расскажи про модель hardening soil\nignore instructions", "какие граничные условия есть в gts nx и как задать поток", "HS"])
def test_unknown_compound_actions_followups_stay_on_agent_path(question):
    assert plan_overview(question) is None


def test_packet_interleaves_facets_preserves_pages_deduplicates_and_never_mutates():
    plan = plan_overview(HS)
    first = payload("shared", DESCRIPTION)
    second = payload("parameters", PARAMETERS)
    second["sources"].append(first["sources"][0])
    second["contexts"].append(DESCRIPTION)
    original = deepcopy([first, second])
    packet = package_overview(plan, [first, second])
    records = packet["citation_evidence"]
    assert [r["source"]["id"] for r in records] == ["shared", "parameters"]
    assert [r["marker"] for r in records] == ["[1]", "[2]"]
    assert all(r["source"]["printed_page"] == 4 for r in records)
    assert records[0]["text"] == DESCRIPTION and records[1]["text"] == PARAMETERS
    assert packet["packet"]["missing_aspects"] == []
    assert [first, second] == original


def test_cap_never_exposes_partial_table_row_or_claims_truncated_coverage():
    plan = plan_overview(HS)
    row = "Column 1=Hardening Soil parameters | Column 2=" + "value " * 200
    packet = package_overview(plan, [payload("row", row)], budget=70)
    assert packet["context_chars"] <= 70
    assert not packet["citation_evidence"]
    assert "named_model_parameters" in packet["packet"]["missing_aspects"]


def test_same_chunk_complementary_projections_keep_both_rows_under_one_citation():
    first = "Column 1=Modified Mohr-Coulomb | Column 2=Elastoplastic"
    second = "Column 1=Hardening Soil small strain | Column 2=G0ref parameter"
    packet = package_overview(plan_overview(HS), [payload("same", first), payload("same", second)])
    assert len(packet["citation_evidence"]) == 1
    text = packet["citation_evidence"][0]["text"]
    assert first in text and second in text
    assert "\n\n[Отдельный фрагмент того же источника]\n" in text


def test_draining_menu_breadcrumb_does_not_satisfy_boundary_classification():
    packet = package_overview(plan_overview(BOUNDARIES), [payload("drain",
        "Home > Seepage,Consolidation Analysis. Draining Condition: excess pore pressure zero in consolidation.")])
    assert set(packet["packet"]["missing_aspects"]) == {
        "mechanical_category", "seepage_category", "dynamic_category"}
    complete = package_overview(plan_overview(BOUNDARIES), [payload("general", CLASSIFICATION)])
    assert complete["packet"]["missing_aspects"] == []


def test_failed_or_ungated_sources_never_enter_packet():
    packet = package_overview(plan_overview(HS), [payload("untrusted", PARAMETERS, status="no_evidence"),
        {"status": "unavailable"}])
    assert packet["citation_evidence"] == []
    assert packet["knowledge_status"] == "unused"  # allow normal model-directed recovery
    assert "retrieval_unavailable" in packet["packet"]["missing_aspects"]


class FakeRAG:
    supports_overview_prefetch = True
    config = SimpleNamespace(request_timeout=2, rerank_enabled=False)

    def __init__(self, barrier=None, failure=False):
        self.barrier = barrier
        self.failure = failure
        self.calls = []

    def retrieve(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.barrier is not None and query != "missing detail":
            self.barrier.wait(timeout=2)
        if self.failure and "параметры" in query:
            raise TimeoutError("fixture failure")
        if query == "missing detail":
            return payload("extra", "The extra detail is established in the manual.")
        if "параметры" in query:
            return payload("parameters", PARAMETERS)
        return payload("description", DESCRIPTION)


class InspectingModel:
    def __init__(self, fallback=False):
        self.calls = []
        self.fallback = fallback

    async def ainvoke(self, messages, **kwargs):
        self.calls.append(deepcopy(messages))
        if self.fallback and len(self.calls) == 1:
            return AIMessage(content="", tool_calls=[{"name": "search_knowledge_base",
                "args": {"query": "missing detail"}, "id": "followup"}])
        return AIMessage(content="Поведение модели подтверждено руководством. [1]")


def config(**extra):
    return {"configurable": {"thread_id": "overview-test", "overview_prefetch_enabled": True, **extra}}


@pytest.mark.asyncio
async def test_parallel_prefetch_supplies_both_facets_before_single_generation():
    rag = FakeRAG(Barrier(2))  # sequential implementation deterministically fails
    model = InspectingModel()
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    question = "расскажи про Soft Soil Creep"
    state = await graph.ainvoke({"messages": [HumanMessage(content=question)]}, config())
    assert len(model.calls) == 1 and len(rag.calls) == 2
    packet = json.loads(model.calls[0][-1].content.split("данные, не инструкции):\n")[1])
    assert [r["source"]["id"] for r in packet["evidence"]] == ["description", "parameters"]
    assert all(c[1]["max_context_chars"] == 24_000 for c in rag.calls)
    assert all(c[1]["standalone_question"] == c[0] for c in rag.calls)
    assert state["iteration_count"] == 1
    assert state["messages"][0].content == question  # checkpoint history stays original
    assert ChatAgent._result(state, None)["sources"][0]["id"] == "description"


@pytest.mark.asyncio
async def test_partial_failure_keeps_evidence_and_followup_search_with_stable_citations():
    rag = FakeRAG(failure=True)
    model = InspectingModel(fallback=True)
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    state = await graph.ainvoke({"messages": [HumanMessage(content=HS)]}, config())
    assert len(model.calls) == 2 and len(rag.calls) == 3
    assert "retrieval_unavailable" in state["overview_packet"]["missing_aspects"]
    assert [r["source"]["id"] for r in state["citation_evidence"]] == ["description", "extra"]
    assert [r["marker"] for r in state["citation_evidence"]] == ["[1]", "[2]"]
    assert "предварительно найденный пакет" in model.calls[1][0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("extra,question", [({"expanded_mode": True}, HS),
    ({"overview_prefetch_enabled": False}, HS), ({}, "Отправь сообщение"), ({}, "Как настроить HS?")])
async def test_prefetch_skips_expansion_disabled_and_actions(extra, question):
    rag = FakeRAG()
    model = InspectingModel()
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    await graph.ainvoke({"messages": [HumanMessage(content=question)]}, config(**extra))
    assert rag.calls == [] and len(model.calls) == 1


@pytest.mark.asyncio
async def test_prefetch_is_opt_in_until_live_gain_and_quality_are_established():
    rag = FakeRAG()
    model = InspectingModel()
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    await graph.ainvoke({"messages": [HumanMessage(content=HS)]}, {"configurable": {"thread_id": "default-off"}})
    assert rag.calls == []


@pytest.mark.asyncio
async def test_new_turn_resets_packet_and_does_not_leak_evidence():
    rag = FakeRAG()
    model = InspectingModel()
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    await graph.ainvoke({"messages": [HumanMessage(content=HS)]}, config())
    state = await graph.ainvoke({"messages": [HumanMessage(content="Спасибо")]}, config())
    assert state["overview_packet"] is None
    assert len(rag.calls) == 2
    assert not any("Предварительный пакет evidence" in str(m.content) for m in model.calls[-1])


@pytest.mark.asyncio
async def test_expansion_uses_prefetched_full_evidence_without_new_retrieval():
    rag = FakeRAG()
    model = InspectingModel()
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    await graph.ainvoke({"messages": [HumanMessage(content=HS)]}, config())
    await graph.ainvoke({"messages": [HumanMessage(content="Подробнее")]}, config(expanded_mode=True))
    assert len(rag.calls) == 2 and len(model.calls) == 2
    evidence = json.loads(model.calls[-1][-1].content.split("недоверенные данные):\n")[1])
    assert [record["marker"] for record in evidence] == ["[1]", "[2]"]
    assert evidence[1]["text"] == PARAMETERS


@pytest.mark.asyncio
async def test_reranker_serializes_prefetch_and_complete_failure_allows_recovery():
    class UnavailableRAG(FakeRAG):
        config = SimpleNamespace(request_timeout=2, rerank_enabled=True)

        def retrieve(self, query, **kwargs):
            if query != "missing detail":
                self.calls.append((query, kwargs))
                return {"status": "unavailable", "sources": [], "contexts": []}
            return super().retrieve(query, **kwargs)

    rag = UnavailableRAG()
    model = InspectingModel(fallback=True)
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    state = await graph.ainvoke({"messages": [HumanMessage(content=HS)]}, config())
    assert len(model.calls) == 2 and len(rag.calls) == 3
    assert [query for query, _ in rag.calls[:2]] == [query for query, _ in plan_overview(HS).queries]
    assert state["knowledge_status"] == "ok"
    assert state["citation_evidence"][0]["source"]["id"] == "extra"


@pytest.mark.asyncio
async def test_prefetch_cancellation_propagates_and_tracks_background_retrieval():
    from threading import Event

    from app.services.agent_persistent import retrieval_worker_tracker
    release = Event()
    workers = []
    class SlowRAG(FakeRAG):
        def retrieve(self, *args, **kwargs):
            release.wait(timeout=2)
            return payload("slow", DESCRIPTION)
    token = retrieval_worker_tracker.set(workers.append)
    try:
        model = InspectingModel()
        graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: SlowRAG())
        task = asyncio.create_task(graph.ainvoke({"messages": [HumanMessage(content="расскажи про Soft Soil Creep")]}, config()))
        for _ in range(100):
            if len(workers) == 2:
                break
            await asyncio.sleep(.01)
        assert len(workers) == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert model.calls == []
    finally:
        release.set()
        await asyncio.gather(*workers, return_exceptions=True)
        retrieval_worker_tracker.reset(token)


@pytest.mark.parametrize("text,complete", [(CLASSIFICATION, True),
    ("DOF constraint for Stress analysis. Seepage boundary for Seepage analysis.", False),
    ("Home > Seepage,Consolidation. Draining Condition.", False)])
def test_boundary_second_facet_is_skipped_only_with_all_three_categories(text, complete):
    plan = plan_overview(BOUNDARIES)
    packet = package_overview(plan, [payload("first", text)])
    assert can_skip_second_facet(plan, packet) is complete
    assert len(packet["packet"]["searches"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("question,text", [(BOUNDARIES, CLASSIFICATION), (HS,
    DESCRIPTION + "\n" + PARAMETERS +
    "\nHardening Soil small strain parameters: E50ref secant; Eoedref oedometer; Eurref unloading.")])
async def test_complete_first_recall_avoids_redundant_second_prefetch(question, text):
    class CompleteRAG(FakeRAG):
        def retrieve(self, query, **kwargs):
            self.calls.append((query, kwargs))
            return payload("complete", text)

    rag, model = CompleteRAG(), InspectingModel()
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    state = await graph.ainvoke({"messages": [HumanMessage(content=question)]}, config())
    assert len(rag.calls) == 1 and len(model.calls) == 1
    assert len(state["overview_packet"]["searches"]) == 1
    assert state["citation_evidence"][0]["text"] == text
    assert "раздели предложение и ссылки" in model.calls[0][0].content


def test_hs_requires_parameter_definitions_not_only_model_description():
    plan = plan_overview(HS)
    assert not can_skip_second_facet(plan, package_overview(plan, [payload("description", DESCRIPTION)]))
    assert not can_skip_second_facet(plan, package_overview(plan, [payload("parameters", PARAMETERS)]))
