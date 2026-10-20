"""Offline, format-neutral evidence preparation and persistent graph contracts."""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from llama_index.core.schema import NodeWithScore, TextNode

from app.chat.agent import ChatAgent
from app.services.agent_persistent import (
    APPROVAL_NODE,
    KNOWLEDGE_UNAVAILABLE,
    build_agent,
    retrieval_worker_tracker,
)
from app.services.prepared_evidence import (
    bounded_evidence,
    prepare_evidence,
    should_prepare_evidence,
)
from app.services.rag import NO_ANSWER, RAGConfig, RAGService


def payload(identity="first", text="The manual establishes this fact.", suffix="pdf", **scope):
    return {"status": "ok", "sources": [{"id": identity, "chunk_id": identity,
        "file_name": f"manual.{suffix}", "product": "gts_nx", "version": "unknown",
        "page": 7 if suffix == "pdf" else None, "language": "unknown", **scope}],
        "contexts": [text]}


def test_model_projection_preserves_literal_scope_and_visual_limitations_on_reuse():
    from app.services.prepared_evidence import evidence_limitations, source_for_model
    source = payload()["sources"][0]
    source.update(block_ids=["retained-in-saved-API-source"], figure_regions=[{"kind": "image"}],
                  source_id="exact-source", location_id="exact-location",
                  block_attributes={"verified": True, "verification_scope": "literal_image_transcription",
                                    "semantic_verified": False})
    compact = source_for_model(source)
    assert source_for_model(compact) == compact
    assert "source_id" not in compact and "location_id" not in compact
    assert compact["block_attributes"]["semantic_verified"] is False
    assert compact["block_attributes"]["verification_scope"] == "literal_image_transcription"
    assert "visual_regions_not_transcribed" in evidence_limitations({}, [compact])
    assert "block_ids" not in compact and source["block_ids"] == ["retained-in-saved-API-source"]


def test_literal_group_model_projection_keeps_constituent_quality_flags_without_geometry_or_duplicate_text():
    from app.services.prepared_evidence import source_for_model
    source = payload()["sources"][0]
    source["block_attributes"] = {"verified": False, "literal_blocks": [
        {"block_id": "first", "kind": "paragraph", "text": "Do not duplicate this text.",
         "bbox": [1, 2, 3, 4], "attributes": {"reading_order": "confirmed", "glyphs": [1]}},
        {"block_id": "second", "kind": "formula_candidate", "attributes": {
            "reading_order": "layout_candidate_unverified", "quality_status": "needs_review"}}]}
    compact = source_for_model(source)
    rows = compact["block_attributes"]["literal_blocks"]
    assert rows[0] == {"block_id": "first", "kind": "paragraph", "attributes": {"reading_order": "confirmed"}}
    assert rows[1]["attributes"] == {"reading_order": "layout_candidate_unverified", "quality_status": "needs_review"}
    assert source_for_model(compact) == compact


class RAG:
    supports_prepared_evidence = True
    config = SimpleNamespace(request_timeout=2)

    def __init__(self, results=None):
        self.calls = []
        self.results = results or [payload()]

    def retrieve(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return deepcopy(self.results[min(len(self.calls) - 1, len(self.results) - 1)])


class Model:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = iter(responses or [AIMessage(content="Подтверждённый факт. [1]")])

    async def ainvoke(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), kwargs))
        return next(self.responses)


def config(**kwargs):
    return {"configurable": {"thread_id": "prepared", **kwargs}}


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    # Tokenizer bootstrap may download its vocabulary on a clean machine.
    # These graph contracts exercise the documented offline fallback instead.
    monkeypatch.setattr("app.chat.service._o200k_base_encoding", lambda: None)


@pytest.mark.parametrize("question", ["Что такое неизвестная функция?", "Как задать нагрузку?",
    "Почему возникает ошибка 105?", "Сравни варианты расчёта", "Какие ограничения есть у решателя?",
    "А как это изменить?", "Расскажи про Hardening Soil small strain", "What is this option?"])
def test_guard_has_no_subject_registry(question):
    assert should_prepare_evidence(question)


@pytest.mark.parametrize("question", ["Привет!", "Спасибо", "Отправь обзор в Telegram",
    "Удалите файл", "Can you send this text?", "Переведи следующий текст: GTS NX",
    "Сократи предыдущий ответ", "Rewrite this message: check the manual"])
def test_guard_skips_conversation_rewrites_and_explicit_effects(question):
    assert not should_prepare_evidence(question)


def test_packet_bounds_before_numbering_and_never_certifies_semantic_sufficiency():
    result = payload(text="Header\nColumn 1=A | Column 2=" + "value " * 40)
    result["sources"].append(payload("second")["sources"][0])
    result["contexts"].append("A later fact.")
    before = deepcopy(result)
    prepared = prepare_evidence(result, query="question", budget=60)
    assert result == before
    records = prepared["citation_evidence"]
    assert [record["marker"] for record in records] == ["[1]", "[2]"]
    assert records[0]["text"] == "Header"
    assert records[1]["text"] == "A later fact."
    assert prepared["packet"]["context_truncated"]
    assert not {"ready", "complete", "coverage_checks"}.intersection(prepared["packet"])
    assert bounded_evidence("Column 1=A | Column 2=Elastic\nColumn 1=B | Column 2=Plastic", 35) == ""


def test_packet_preserves_scopes_with_colliding_chunk_ids():
    first = payload(text="First version.", version="2021")
    second = payload(text="Second version.", version="2024")
    result = {"status": "ok", "sources": first["sources"] + second["sources"],
              "contexts": first["contexts"] + second["contexts"]}
    assert len(prepare_evidence(result, query="compare")["citation_evidence"]) == 2


def test_model_metadata_projection_bounds_geometry_without_changing_evidence_or_scope():
    result = payload(text="x = 1", block_attributes={"visual_review_required": True, "verified": False,
        "glyphs": [{"text": "x", "bbox": [1, 2, 3, 4], "font": "a" * 30}] * 1000,
        "raw_char_text": "x" * 10000, "cells": ["x"] * 1000},
        figure_regions=[{"bbox": [1, 2, 3, 4], "image_sha256": "a" * 64}] * 100,
        source_hash="b" * 64, source_locator="manual.pdf#page=7")
    before = deepcopy(result)
    prepared = prepare_evidence(result, query="equation")
    record = prepared["citation_evidence"][0]
    assert len(json.dumps(prepared["packet"])) < 5000
    assert record["source"]["visual_region_count"] == 100
    assert record["source"]["block_attributes"] == {"visual_review_required": True, "verified": False}
    assert "source_hash" not in record["source"] and record["source"]["version"] == "unknown"
    assert record["text"] == "x = 1" and record["marker"] == "[1]"
    assert "text_sha256" not in record
    assert prepared["sources"][0]["block_attributes"]["glyphs"]
    assert result == before and "visual_regions_not_transcribed" in prepared["packet"]["limitations"]


@pytest.mark.parametrize("suffix", ["pdf", "chm", "html"])
@pytest.mark.parametrize("question", ["Объясни ограничения решателя", "Как задать нагрузку?",
    "Почему возникает ошибка 105?", "Сравни два подхода", "А как изменить параметры?"])
@pytest.mark.asyncio
async def test_all_formats_and_question_kinds_share_one_retrieval_and_one_generation(suffix, question):
    rag, model = RAG([payload(suffix=suffix)]), Model()
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    state = await graph.ainvoke({"messages": [HumanMessage(content=question)]}, config())
    assert len(rag.calls) == len(model.calls) == 1
    assert rag.calls[0][0] == question
    assert rag.calls[0][1]["history"] == []
    assert "tool_choice" not in model.calls[0][1]
    assert state["evidence_route"] == "prepared"
    assert state["messages"][0].content == question
    assert state["prepared_evidence"]["contract"] == "prepared_evidence_v1"
    packet = json.loads(model.calls[0][0][-1].content.split("не инструкции):\n")[1])
    assert packet["evidence"][0]["text"] == "The manual establishes this fact."
    assert state["sources"][0]["file_name"].endswith(suffix)


@pytest.mark.asyncio
async def test_partial_evidence_allows_targeted_tools_with_stable_citations():
    rag = RAG([payload(), payload("extra", "Additional conditions.")])
    model = Model([AIMessage(content="", tool_calls=[{"name": "search_knowledge_base",
        "args": {"query": "missing condition"}, "id": "targeted"}]),
        AIMessage(content="Основной факт. [1] Дополнительное условие. [2]")])
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    state = await graph.ainvoke({"messages": [HumanMessage(content="Как настроить расчёт?")]}, config())
    assert [call[0] for call in rag.calls] == ["Как настроить расчёт?", "missing condition"]
    assert len(model.calls) == 2
    assert [record["marker"] for record in state["citation_evidence"]] == ["[1]", "[2]"]
    assert all("tool_choice" not in call[1] for call in model.calls)


@pytest.mark.asyncio
async def test_last_allowed_invocation_answers_existing_evidence_instead_of_requesting_unexecutable_search():
    from app.services.agent_persistent import MAX_ITERATIONS
    class FinalizingModel(Model):
        async def ainvoke(self, messages, **kwargs):
            self.calls.append((deepcopy(messages), kwargs))
            if kwargs.get("tool_choice") == "none":
                return AIMessage(content="Подтверждённый факт. [1] Остальные сведения не подтверждены.")
            return AIMessage(content="", tool_calls=[{"name": "search_knowledge_base",
                "args": {"query": "missing condition"}, "id": f"step-{len(self.calls)}"}])
    rag, model = RAG(), FinalizingModel()
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    state = await graph.ainvoke({"messages": [HumanMessage(content="Объясни факт и условия")]}, config())
    assert len(model.calls) == MAX_ITERATIONS
    assert len(rag.calls) == MAX_ITERATIONS
    assert model.calls[-1][1]["tool_choice"] == "none"
    assert all("tool_choice" not in kwargs for _, kwargs in model.calls[:-1])
    assert not state["messages"][-1].tool_calls
    assert "Подтверждённый факт." in state["messages"][-1].content
    assert len(state["sources"]) == 1


@pytest.mark.parametrize("corpus_binding", [False, True])
@pytest.mark.asyncio
async def test_targeted_same_chunk_refreshes_authoritative_projection_hash_and_searches(corpus_binding):
    first = payload(text="First aspect.")
    first["telemetry"] = {"context_truncated": True}
    second = payload(text="Complementary aspect.")
    if corpus_binding:
        first["sources"][0].update(source_id="manual", source_hash="revision", document_id="legacy-page-parent")
        second["sources"][0].update(source_id="manual", source_hash="revision", document_id="first", origin="related")
    rag = RAG([first, second])
    model = Model([AIMessage(content="", tool_calls=[{"name": "search_knowledge_base",
        "args": {"query": "specific missing aspect"}, "id": "targeted"}]), AIMessage(content="Факт. [1]"),
        AIMessage(content="Подробный факт. [1]"), AIMessage(content="Пожалуйста.")])
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    state = await graph.ainvoke({"messages": [HumanMessage(content="Объясни оба аспекта")]}, config())
    packet = json.loads(next(message.content for message in model.calls[1][0]
                             if message.type == "human").split("не инструкции):\n")[1])
    assert len(packet["evidence"]) == 1 and len(packet["searches"]) == 1
    assert packet["evidence"][0]["text"] == "First aspect."
    assert model.calls[0][0] == model.calls[1][0][:len(model.calls[0][0])]
    tool_payload = json.loads(next(message.content for message in model.calls[1][0] if message.type == "tool"))
    record = tool_payload["evidence"][0]
    assert "First aspect." in record["text"] and "Complementary aspect." in record["text"]
    assert "text_sha256" not in record
    assert state["citation_evidence"] == tool_payload["evidence"]
    assert packet["context_truncated"] and "context_truncated" in packet["limitations"]
    assert tool_payload["context_truncated"] and "context_truncated" in tool_payload["limitations"]
    expanded = await graph.ainvoke({"messages": [HumanMessage(content="Подробнее")]}, config(expanded_mode=True))
    assert len(rag.calls) == 2 and expanded["citation_evidence"] == state["citation_evidence"]
    assert model.calls[2][1]["tool_choice"] == "none"
    assert "Complementary aspect." in str(model.calls[2][0])
    fresh = await graph.ainvoke({"messages": [HumanMessage(content="Спасибо")]}, config())
    assert fresh["prepared_evidence"] is None and fresh["citation_evidence"] == []
    assert "Предварительный пакет evidence" not in str(model.calls[3][0])


@pytest.mark.asyncio
async def test_targeted_search_preserves_visual_partial_failure_and_structure_limits():
    extra = payload("extra", "Equation candidate.", block_attributes={"visual_review_required": True},
                    figure_regions=[{"bbox": [1, 2, 3, 4]}])
    extra["telemetry"] = {"retrieval_partial_failure": True, "related_evidence_truncated": True}
    rag = RAG([payload(), extra])
    model = Model([AIMessage(content="", tool_calls=[{"name": "search_knowledge_base",
        "args": {"query": "missing formula"}, "id": "targeted"}]), AIMessage(content="Факт. [1]")])
    state = await build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag).ainvoke(
        {"messages": [HumanMessage(content="Объясни формулу")]}, config())
    assert {"retrieval_partial_failure", "structural_expansion_truncated", "unverified_formula_or_layout",
            "visual_regions_not_transcribed"}.issubset(state["prepared_evidence"]["limitations"])
    update = json.loads(next(m.content for m in model.calls[1][0] if m.type == "tool"))
    assert update["limitations"] == state["prepared_evidence"]["limitations"]
    assert update["evidence"][0]["source"]["block_attributes"]["visual_review_required"]


@pytest.mark.asyncio
async def test_append_evidence_overflow_compacts_current_invoke_without_dropping_current_records(monkeypatch):
    def tokens(messages):
        # The append log exceeds the limit; the cumulative representation fits.
        return 20 if any('"evidence":' in m["content"] and m["role"] == "tool" for m in messages) else 10
    monkeypatch.setattr("app.chat.service.count_tokens", tokens)
    rag = RAG([payload(), payload("extra", "Additional condition.")])
    model = Model([AIMessage(content="", tool_calls=[{"name":"search_knowledge_base",
        "args":{"query":"condition"},"id":"extra"}]), AIMessage(content="Факт. [1] Условие. [2]")])
    state = await build_agent(InMemorySaver(),model=model,rag_provider=lambda:rag).ainvoke(
        {"messages":[HumanMessage(content="Факт и условия?")]},config(context_budget=15))
    messages = model.calls[1][0]
    packet = json.loads(next(m.content for m in messages if m.type=="human").split("не инструкции):\n")[1])
    assert packet["evidence"] == state["citation_evidence"]
    assert len(packet["evidence"]) == 2 and len(packet["searches"]) == 2
    tool = json.loads(next(m.content for m in messages if m.type=="tool"))
    assert "evidence" not in tool and tool["evidence_location"] == "current prepared packet"


@pytest.mark.parametrize("status,answer", [("no_evidence", NO_ANSWER), ("unavailable", KNOWLEDGE_UNAVAILABLE)])
@pytest.mark.asyncio
async def test_miss_and_outage_are_distinct_and_tools_remain_available(status, answer):
    rag, model = RAG([{"status": status}]), Model([AIMessage(content="Не удалось найти данные.")])
    state = await build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag).ainvoke(
        {"messages": [HumanMessage(content="Как устроена функция?")]}, config())
    assert state["knowledge_status"] == status
    assert state["messages"][-1].content == answer
    assert "tool_choice" not in model.calls[0][1]


@pytest.mark.asyncio
async def test_outage_can_recover_through_model_directed_search():
    rag = RAG([{"status": "unavailable"}, payload("recovered")])
    model = Model([AIMessage(content="", tool_calls=[{"name": "search_knowledge_base",
        "args": {"query": "specific detail"}, "id": "recovery"}]), AIMessage(content="Факт. [1]")])
    state = await build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag).ainvoke(
        {"messages": [HumanMessage(content="Объясни настройку")]}, config())
    assert state["knowledge_status"] == "ok" and len(state["sources"]) == 1


@pytest.mark.parametrize("old_flag", ["overview_prefetch_enabled", "verified_overview_enabled"])
@pytest.mark.asyncio
async def test_old_flags_cannot_enable_topic_specific_routes(old_flag):
    rag, model = RAG(), Model()
    state = await build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag).ainvoke(
        {"messages": [HumanMessage(content="Расскажи про Hardening Soil small strain")]},
        config(prepared_evidence_enabled=False, **{old_flag: True}))
    assert rag.calls == [] and "tool_choice" not in model.calls[0][1]
    assert state["prepared_evidence"] is None
    assert not any("G0ref" in str(message.content) for message in model.calls[0][0])


@pytest.mark.asyncio
async def test_expansion_reuses_packet_and_new_turn_clears_evidence():
    rag = RAG()
    model = Model([AIMessage(content="Факт. [1]"), AIMessage(content="Подробный факт. [1]"), AIMessage(content="Пожалуйста.")])
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    first = await graph.ainvoke({"messages": [HumanMessage(content="Объясни функцию")]}, config())
    expanded = await graph.ainvoke({"messages": [HumanMessage(content="Подробнее")]}, config(expanded_mode=True))
    assert len(rag.calls) == 1 and expanded["citation_evidence"] == first["citation_evidence"]
    assert model.calls[1][1]["tool_choice"] == "none"
    state = await graph.ainvoke({"messages": [HumanMessage(content="Спасибо")]}, config())
    assert state["prepared_evidence"] is None and not state["citation_evidence"]
    assert "Предварительный пакет evidence" not in str(model.calls[2][0])


@pytest.mark.asyncio
async def test_explicit_delivery_skips_prefetch_and_still_requires_approval():
    rag, send = RAG(), AsyncMock()
    model = Model([AIMessage(content="", tool_calls=[{"name": "send_telegram_message",
        "args": {"chat_id": "123", "text": "Обзор"}, "id": "send"}])])
    graph = build_agent(InMemorySaver(), model=model, send_message=send, rag_provider=lambda: rag)
    await graph.ainvoke({"messages": [HumanMessage(content="Отправь обзор в Telegram")]},
                        config(user_role="write-with-approve"))
    assert rag.calls == []
    assert APPROVAL_NODE in (await graph.aget_state(config())).next
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_tracks_worker_until_blocking_retrieval_finishes():
    release, workers = Event(), []
    class SlowRAG(RAG):
        def retrieve(self, *args, **kwargs):
            release.wait(timeout=2)
            return payload()
    token = retrieval_worker_tracker.set(workers.append)
    try:
        model = Model()
        graph = build_agent(InMemorySaver(), model=model, rag_provider=SlowRAG)
        task = asyncio.create_task(graph.ainvoke({"messages": [HumanMessage(content="Объясни функцию")]}, config()))
        for _ in range(100):
            if workers:
                break
            await asyncio.sleep(.01)
        assert len(workers) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert model.calls == []
    finally:
        release.set()
        await asyncio.gather(*workers, return_exceptions=True)
        retrieval_worker_tracker.reset(token)


@pytest.mark.asyncio
async def test_new_setting_defaults_on_and_chat_propagates_it():
    graph = SimpleNamespace(ainvoke=AsyncMock(return_value={}))
    chat = SimpleNamespace(interface="test", owner_external_id="1", id="chat")
    adapter = ChatAgent(graph, None)
    await adapter._invoke({}, chat)
    assert graph.ainvoke.call_args.args[1]["configurable"]["prepared_evidence_enabled"]
    adapter.settings = SimpleNamespace(agent_prepared_evidence_enabled=False,
        chat_model_context_tokens=10000, chat_safety_margin_tokens=100, chat_context_window=10)
    await adapter._invoke({}, chat)
    assert not graph.ainvoke.call_args.args[1]["configurable"]["prepared_evidence_enabled"]
    assert not RAGConfig.from_settings(SimpleNamespace(agent_verified_overview_enabled=True)).evidence_scope_enabled


def test_sources_keep_relative_locators_unknown_version_and_pdf_lines(tmp_path):
    service = RAGService(replace(RAGConfig.from_settings(SimpleNamespace()), data_dir=tmp_path))
    for suffix in ("pdf", "html", "chm"):
        metadata = {"file_name": "manual." + suffix, "file_path": str(tmp_path / "nested" / ("manual." + suffix)),
                    "product": "gts_nx", "page": 3 if suffix == "pdf" else None}
        if suffix == "chm":
            metadata["chm_member"] = "/help/options.htm"
        item = NodeWithScore(node=TextNode(text="Heading\nA table row\nOther row", metadata=metadata), score=.9)
        source = service._source_from_node(item, include_snippet=False)
        assert source["version"] == source["language"] == source["section"] == "unknown"
        assert source["product"] == "gts_nx" and source["source_locator"].startswith("nested/manual.")
        assert str(tmp_path) not in source["source_locator"]
        if suffix != "chm":
            assert service._context_text(item).count("\n") == 2


def test_dedup_only_collapses_exact_cross_format_content_with_confirmed_origin(tmp_path):
    service = RAGService(replace(RAGConfig.from_settings(SimpleNamespace()), data_dir=tmp_path))
    def node(identity, suffix, product="gts_nx", version="unknown", text="Exact original content.", origin="source-original"):
        return NodeWithScore(node=TextNode(id_=identity, text=text, metadata={"file_name": "manual." + suffix,
            "product": product, "version": version, "parent_source_id": origin}), score=.9)
    items = [node("pdf", "pdf"), node("html", "html"), node("civil", "html", product="civil"),
             node("old", "chm", version="2021"), node("unrelated", "html", origin="different-source"),
             node("changed", "html", text="Exact original content. Changed applicability.")]
    assert [item.node.node_id for item in service._deduplicate(items, limit=10)] == ["pdf", "civil", "old", "unrelated", "changed"]


@pytest.mark.asyncio
async def test_initial_graph_preserves_raw_visual_provenance_until_single_model_projection():
    result = payload(text="A visible native fragment.",
        block_attributes={"visual_review_required": True, "verified": False,
                          "glyphs": [{"text": "x", "bbox": [1, 2, 3, 4]}]},
        figure_regions=[{"kind": "image", "bbox": [1, 2, 3, 4], "image_sha256": "image"}],
        source_hash="source", source_locator="manual.pdf#page=7")
    rag, model = RAG([result]), Model()
    state = await build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag).ainvoke(
        {"messages": [HumanMessage(content="Объясни параметр по документации")]}, config())
    packet = json.loads(model.calls[0][0][-1].content.split("не инструкции):\n")[1])
    source = packet["evidence"][0]["source"]
    assert source["visual_region_count"] == 1 and source["untranscribed_visual_regions_present"] is True
    assert {"visual_regions_not_transcribed", "unverified_formula_or_layout"} <= set(packet["limitations"])
    assert "glyphs" not in source["block_attributes"]
    assert state["sources"][0]["block_attributes"]["glyphs"]
    assert state["sources"][0]["figure_regions"] == result["sources"][0]["figure_regions"]
    assert source["source_locator"] == "manual.pdf#page=7"
    assert packet["evidence"][0]["text"] == result["contexts"][0]
    assert len(rag.calls) == len(model.calls) == 1



@pytest.mark.asyncio
async def test_prepared_retrieval_keeps_previous_dialogue_without_current_query_duplicate():
    rag, model = RAG(), Model()
    messages = [HumanMessage(content="Ранее обсуждали расчёт"), AIMessage(content="Предыдущий ответ"),
                HumanMessage(content="А как изменить параметры?")]
    await build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag).ainvoke(
        {"messages": messages}, config())
    assert rag.calls[0][1]["history"] == [
        {"role": "user", "content": "Ранее обсуждали расчёт"},
        {"role": "assistant", "content": "Предыдущий ответ"}]
