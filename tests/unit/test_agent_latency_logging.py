"""Latency diagnostics must cover actual calls without logging their content."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from llama_index.core.schema import NodeWithScore, TextNode

import app.services.agent_persistent as persistent
from app.services import rag


def test_retrieval_logs_stage_times_and_actual_attempts_without_content(monkeypatch):
    question = "Как настроить процедуру?"
    translation = "How do I configure this calculation procedure?"
    evidence = "Confidential manual fragment."
    source = NodeWithScore(
        node=TextNode(id_="fixture", text=evidence, metadata={"source": "manual.pdf", "page_number": 7}),
        score=0.95,
    )
    service = rag.RAGService(rag.RAGConfig.from_settings(SimpleNamespace(rag_allow_external_data_transfer=True)))
    dense_queries, lexical_queries, events = [], [], []

    def dense(query):
        dense_queries.append(query)
        return [source]

    def lexical(query):
        lexical_queries.append(query)
        return [source]  # Every bounded variant runs; duplicate evidence is merged.

    service._retriever = SimpleNamespace(retrieve=dense)
    service._lexical_available = True
    monkeypatch.setattr(service, "_lexical_queries", lambda *_args: ["first", "unused"])
    monkeypatch.setattr(service, "_retrieve_lexical_candidates", lexical)
    monkeypatch.setattr(rag, "logger", SimpleNamespace(info=lambda event, **fields: events.append((event, fields))))

    result = service.retrieve(question, standalone_question=question, translated_question=translation)

    assert dense_queries == [question, translation]
    # The translated question also joins the lexical variants so English
    # manuals get the literal-match path (block 2.4).
    assert lexical_queries == ["first", "unused", translation]
    assert result["status"] == "ok" and result["sources"][0]["page"] == 7
    assert len(events) == 1
    event, fields = events[0]
    assert event == "rag_retrieval_completed" and fields["outcome"] == "evidence_ready"
    assert fields["dense_query_count"] == 2
    assert fields["lexical_attempt_count"] == 3
    assert fields["lexical_query_variant_count"] == 3
    for key in ("dense_retrieval_ms", "lexical_retrieval_ms", "selection_ms", "retrieval_ms"):
        assert isinstance(fields[key], (int, float)) and fields[key] >= 0
    serialized = json.dumps(events, ensure_ascii=False)
    assert all(content not in serialized for content in (question, translation, evidence))


@pytest.mark.asyncio
@pytest.mark.parametrize("expanded", [False, True])
async def test_model_logs_elapsed_time_and_mode_without_content(monkeypatch, expanded):
    events = []
    answer = "Confidential response."

    class Model:
        async def ainvoke(self, _messages, **_kwargs):
            return AIMessage(content=answer, response_metadata={"finish_reason": "stop"})

    monkeypatch.setattr(persistent, "logger", SimpleNamespace(info=lambda event, **fields: events.append((event, fields))))
    graph = persistent.build_agent(InMemorySaver(), model=Model())
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Confidential question.")]},
        {"configurable": {"thread_id": "latency", "expanded_mode": expanded}},
    )

    assert result["messages"][-1].content == answer
    fields = next(fields for event, fields in events if event == "agent_model_usage")
    assert fields["mode"] == ("expanded" if expanded else "brief")
    assert fields["iteration"] == 1 and fields["tool_call_count"] == 0
    assert isinstance(fields["model_ms"], (int, float)) and fields["model_ms"] >= 0
    assert "Confidential" not in json.dumps(events)
