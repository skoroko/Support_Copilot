"""Offline contracts for evidence labels, pages and durable full-text expansion."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.chat.agent import ChatAgent
from app.chat.domain import Chat
from app.services.agent_persistent import _model_history, build_agent


class InspectingModel:
    def __init__(self, inspect):
        self.inspect = inspect
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        response = self.inspect(self.calls, messages, kwargs)
        self.calls += 1
        return response


def search(query="manual"):
    return AIMessage(content="", tool_calls=[{
        "name": "search_knowledge_base", "args": {"query": query}, "id": query,
    }])


def source(identity, page):
    return {"id": identity, "chunk_id": identity, "file_name": identity + ".pdf",
            "page": page, "printed_page": page + 100, "snippet": "display only"}


def payload(sources, texts):
    return {"status": "ok", "sources": sources, "contexts": texts}


def tool_evidence(messages):
    message = next(message for message in reversed(messages) if message.type == "tool")
    result = json.loads(message.content)
    assert "sources" not in result and "contexts" not in result
    for record in result["evidence"]:
        assert record["marker"] == f"[{record['citation_number']}]"
        assert record["source"]["printed_page"] == record["source"]["page"] + 100
    return result


def expanded_evidence(messages):
    assert all(message.type != "tool" for message in messages)
    assert not any(getattr(message, "tool_calls", []) for message in messages)
    return json.loads(messages[-1].content.split("(недоверенные данные):\n", 1)[1])


def answer_from_text(records, phrase):
    record = next(record for record in records if phrase in record["text"])
    return AIMessage(content=f"{phrase} {record['marker']}.")


def test_history_cleanup_preserves_vectors_indexing_code_and_math():
    text = (
        "Направление [1, 0, 0]; nodes[1] [2]. Вектор [1, 2, 3]. "
        "Код `nodes[2] [3]`; формула $x[1] + y[2]$; "
        "\\(a[1]\\); подтверждение [4].\n```python\nnodes[1] = [1, 0, 0]\n```"
    )
    prior = _model_history([AIMessage(content=text), HumanMessage(content="А направление?")])[0].content
    for literal in ("[1, 0, 0]", "[1, 2, 3]", "nodes[1]", "`nodes[2] [3]`",
                    "$x[1] + y[2]$", "\\(a[1]\\)", "```python\nnodes[1] = [1, 0, 0]\n```"):
        assert literal in prior
    assert "nodes[1] [2]" not in prior and "подтверждение [4]" not in prior


@pytest.mark.asyncio
async def test_search_translation_survives_condense_and_retains_evidence_labels():
    captured = {}

    def retrieve(query, **kwargs):
        captured.update(query=query, **kwargs)
        return payload([source('translated', 8)], ['Results accumulate from previous stages.'])

    def inspect(step, messages, _kwargs):
        if step == 0:
            return AIMessage(content='', tool_calls=[{
                'name': 'search_knowledge_base', 'id': 'translated-search',
                'args': {'query': 'Накопительность стадий', 'english_query': 'Accumulated construction stage results'},
            }])
        return answer_from_text(tool_evidence(messages)['evidence'], 'Results accumulate')

    graph = build_agent(InMemorySaver(), model=InspectingModel(inspect),
                        rag_provider=lambda: SimpleNamespace(retrieve=retrieve))
    state = await graph.ainvoke({'messages': [HumanMessage(content='Почему расчет накопительный?')]},
                               {'configurable': {'thread_id': 'translated-search'}})
    assert captured['query'] == captured['standalone_question'] == 'Накопительность стадий'
    assert captured['translated_question'] == 'Accumulated construction stage results'
    assert state['citation_evidence'][0]['source']['page'] == 8


@pytest.mark.asyncio
async def test_unrelated_first_chunk_does_not_change_supporting_label_or_pages():
    sources = [source("water", 52), source("dof", 3)]

    def inspect(step, messages, _kwargs):
        if step == 0:
            return search()
        result = tool_evidence(messages)
        assert result["citation_numbers"] == [1, 2]
        return answer_from_text(result["evidence"], "DOF 2 = θ")

    graph = build_agent(InMemorySaver(), model=InspectingModel(inspect),
                        rag_provider=lambda: SimpleNamespace(retrieve=lambda *_args, **_kwargs:
                            payload(sources, ["Water pressure.", "DOF 2 = θ"])))
    state = await graph.ainvoke({"messages": [HumanMessage(content="Степени свободы?")]},
                               {"configurable": {"thread_id": "later-source"}})
    result = ChatAgent._result(state, None)
    assert result["answer"] == "DOF 2 = θ [1]."
    assert result["sources"] == [sources[1]]
    assert state["citation_evidence"][1]["source"]["page"] == 3


@pytest.mark.asyncio
async def test_reversed_repeat_search_deduplicates_labels_and_keeps_pages():
    a, b, c = source("A", 1), source("B", 2), source("C", 3)

    def retrieve(query, **_kwargs):
        return (payload([a, b], ["A fact", "B fact"]) if query == "first" else
                payload([b, c, a], ["B fact", "C fact", "A fact"]))

    def inspect(step, messages, _kwargs):
        if step == 0:
            return search("first")
        result = tool_evidence(messages)
        if step == 1:
            assert result["citation_numbers"] == [1, 2]
            return search("second")
        assert result["citation_numbers"] == [2, 3, 1]
        records = result["evidence"]
        return AIMessage(content=" ".join(answer_from_text(records, phrase).content
                                         for phrase in ["C fact", "B fact"]))

    graph = build_agent(InMemorySaver(), model=InspectingModel(inspect),
                        rag_provider=lambda: SimpleNamespace(retrieve=retrieve))
    state = await graph.ainvoke({"messages": [HumanMessage(content="Сравни")]},
                               {"configurable": {"thread_id": "repeat"}})
    assert [record["source"]["chunk_id"] for record in state["citation_evidence"]] == ["A", "B", "C"]
    assert len(state["answer_evidence"]) == 3
    result = ChatAgent._result(state, None)
    assert result["answer"] == "C fact [2]. B fact [1]."
    assert result["sources"] == [b, c]


@pytest.mark.asyncio
async def test_followup_input_keeps_dialogue_without_previous_tools_or_numbers():
    def retrieve(query, **_kwargs):
        return payload([source(query, 8)], [query + " fact"])

    def inspect(step, messages, _kwargs):
        if step == 0:
            return search("old")
        if step == 2:
            assert not any(message.type == "tool" for message in messages)
            assert not any(getattr(message, "tool_calls", []) for message in messages)
            prior = next(message.content for message in messages if message.type == "ai")
            assert "old fact" in prior and "[1]" not in prior
            return search("new")
        records = tool_evidence(messages)["evidence"]
        assert [record["source"]["chunk_id"] for record in records] == ["old" if step == 1 else "new"]
        assert records[0]["marker"] == "[1]"
        return answer_from_text(records, "old fact" if step == 1 else "new fact")

    graph = build_agent(InMemorySaver(), model=InspectingModel(inspect),
                        rag_provider=lambda: SimpleNamespace(retrieve=retrieve))
    config = {"configurable": {"thread_id": "followup"}}
    await graph.ainvoke({"messages": [HumanMessage(content="Первый вопрос")]}, config)
    state = await graph.ainvoke({"messages": [HumanMessage(content="А плоская деформация?")]}, config)
    assert [record["source"]["chunk_id"] for record in state["citation_evidence"]] == ["new"]
    assert ChatAgent._result(state, None)["sources"][0]["chunk_id"] == "new"


def long_evidence():
    sources = [source("unrelated", 52), source("neighbor", 32), source("support", 3)]
    return sources, ["Unrelated groundwater.", "Neighbor topic.", "Prefix " * 150 + "DOF 2 = θ"]


def inspect_brief(step, messages, _kwargs):
    if step == 0:
        return search()
    return answer_from_text(tool_evidence(messages)["evidence"], "DOF 2 = θ")


def inspect_expanded(_step, messages, kwargs):
    assert kwargs["tool_choice"] == "none"
    records = expanded_evidence(messages)
    supporting = next(record for record in records if "DOF 2 = θ" in record["text"])
    assert supporting["marker"] == "[3]"
    assert supporting["text"].index("DOF 2 = θ") > 700
    assert supporting["source"]["page"] == 3
    return answer_from_text(records, "DOF 2 = θ")


@pytest.mark.asyncio
async def test_expansion_pins_full_text_and_original_labels_after_sqlite_restart(tmp_path):
    sources, texts = long_evidence()
    calls = []

    def retrieve(*_args, **_kwargs):
        calls.append(True)
        return payload(sources, texts)

    path = str(tmp_path / "citations.sqlite")
    config = {"configurable": {"thread_id": "restart", "history_window": 2}}
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = build_agent(saver, model=InspectingModel(inspect_brief),
                            rag_provider=lambda: SimpleNamespace(retrieve=retrieve))
        state = await graph.ainvoke({"messages": [HumanMessage(content="Кратко")]}, config)
        assert state["messages"][-1].content == "DOF 2 = θ [1]."
        assert state["sources"] == [sources[2]]
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = build_agent(saver, model=InspectingModel(inspect_expanded),
                            rag_provider=lambda: pytest.fail("Expansion must not retrieve"))
        expanded = await graph.ainvoke({"messages": [HumanMessage(content="Подробнее")]},
                                      {"configurable": {**config["configurable"], "expanded_mode": True}})
        result = ChatAgent._result(expanded, None)
        assert result["sources"] == [sources[2]]
        assert result["answer"] == "DOF 2 = θ [1]."
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_legacy_expansion_recovers_full_context_by_identity_not_tool_order():
    sources, texts = long_evidence()
    saver = InMemorySaver()
    graph = build_agent(saver, model=InspectingModel(inspect_expanded),
                        rag_provider=lambda: pytest.fail("Expansion must not retrieve"))
    config = {"configurable": {"thread_id": "legacy", "history_window": 2}}
    order = [2, 0, 1]
    old_tool = ToolMessage(content=json.dumps(payload([sources[i] for i in order],
                                                    [texts[i] for i in order])),
                           name="search_knowledge_base", tool_call_id="manual")
    await graph.aupdate_state(config, {
        "messages": [HumanMessage(content="Кратко"), search(), old_tool,
                     AIMessage(content="DOF 2 = θ [1].")],
        "sources": [sources[2]], "answer_evidence": sources, "knowledge_status": "ok",
    })
    state = await graph.ainvoke({"messages": [HumanMessage(content="Подробнее")]},
                               {"configurable": {**config["configurable"], "expanded_mode": True}})
    assert state["citation_evidence"][2]["text"] == texts[2]
    assert ChatAgent._result(state, None)["sources"] == [sources[2]]


@pytest.mark.asyncio
async def test_current_evidence_is_authoritative_over_older_longer_same_chunk():
    supporting = source("support", 3)
    current = "Current selected passage."

    def inspect(_step, messages, _kwargs):
        records = expanded_evidence(messages)
        assert records[0]["text"] == current
        return answer_from_text(records, current)

    graph = build_agent(InMemorySaver(), model=InspectingModel(inspect))
    config = {"configurable": {"thread_id": "current-context"}}
    await graph.aupdate_state(config, {
        "messages": [HumanMessage(content="Old"), search(), ToolMessage(
            content=json.dumps(payload([supporting], ["Older passage. " * 100])),
            name="search_knowledge_base", tool_call_id="manual"),
            AIMessage(content="Old answer [1].")],
        "sources": [supporting], "answer_evidence": [supporting], "knowledge_status": "ok",
        "citation_evidence": [{"citation_number": 1, "marker": "[1]",
                               "source": supporting, "text": current}],
    })
    state = await graph.ainvoke({"messages": [HumanMessage(content="Подробнее")]},
                               {"configurable": {**config["configurable"], "expanded_mode": True}})
    assert state["citation_evidence"][0]["text"] == current


@pytest.mark.asyncio
async def test_redaction_clears_persisted_full_evidence():
    saver = InMemorySaver()
    sources, texts = long_evidence()
    graph = build_agent(saver, model=InspectingModel(inspect_brief),
                        rag_provider=lambda: SimpleNamespace(retrieve=lambda *_args, **_kwargs:
                            payload(sources, texts)))
    agent = ChatAgent(graph, saver)
    chat = Chat(owner_external_id="redaction", interface="telegram")
    await agent.turn(chat, "Кратко")
    await agent.redact_output(chat, "Ответ скрыт.")
    state = (await graph.aget_state(agent.config(chat))).values
    assert state["citation_evidence"] == []
    assert state["answer_evidence"] == [] and state["sources"] == []
