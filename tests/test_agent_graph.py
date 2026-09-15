from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, messages_to_dict

import app.services.agent_graph as agent
from scripts import bench_agents
from scripts.run_agent_comparison import CASES


class _FakeBoundModel:
    """Tool-capable offline model double for both graph styles."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = iter(responses)
        self.received: list[list[Any]] = []
        self.bound_tools: list[Any] | None = None

    def bind_tools(self, tools, **_kwargs):
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages, **_kwargs) -> AIMessage:
        self.received.append(list(messages))
        return next(self._responses)

    def invoke(self, messages, **_kwargs) -> AIMessage:
        self.received.append(list(messages))
        return next(self._responses)


def _tool_call(name: str = "get_current_time", args: dict[str, Any] | None = None, call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="Вызову безопасный локальный инструмент.", tool_calls=[{"name": name, "args": args or {"timezone": "Europe/Moscow"}, "id": call_id}])


@pytest.mark.asyncio
async def test_custom_graph_runs_tool_then_returns_final_answer(monkeypatch):
    fake = _FakeBoundModel([_tool_call(), AIMessage(content="Сейчас время получено локально.")])
    monkeypatch.setattr(agent, "bound_model", fake)

    result = await agent.custom_graph.ainvoke(
        {"messages": [HumanMessage(content="Сколько времени в Москве?")], "iteration_count": 0, "tool_results": []}
    )

    assert result["iteration_count"] == 2
    assert result["tool_results"][0]["tool_name"] == "get_current_time"
    assert any(isinstance(message, ToolMessage) for message in result["messages"])
    assert result["messages"][-1].content == "Сейчас время получено локально."
    assert len(fake.received) == 2
    assert isinstance(fake.received[0][0], SystemMessage)
    assert fake.received[0][0].content == agent.SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_prebuilt_graph_runs_same_no_tool_task_with_tool_capable_fake():
    fake = _FakeBoundModel([AIMessage(content="Нужна проверка инженером поддержки.")])
    graph = agent.build_prebuilt_graph(fake)

    result = await graph.ainvoke({"messages": [HumanMessage(content="Нужна ли эскалация?")]})

    assert result["messages"][-1].content == "Нужна проверка инженером поддержки."
    assert fake.bound_tools is not None
    assert {tool.name for tool in fake.bound_tools} == {"search_knowledge_base", "get_current_time", "send_telegram_message"}


@pytest.mark.asyncio
async def test_both_graphs_answer_the_same_task_under_the_same_system_prompt(monkeypatch):
    answer = "Для точного ответа нужна проверка инженером поддержки."
    custom_fake = _FakeBoundModel([AIMessage(content=answer)])
    prebuilt_fake = _FakeBoundModel([AIMessage(content=answer)])
    monkeypatch.setattr(agent, "bound_model", custom_fake)

    custom = await agent.custom_graph.ainvoke(
        {"messages": [HumanMessage(content="Нужна ли эскалация?")], "iteration_count": 0, "tool_results": []}
    )
    prebuilt = await agent.build_prebuilt_graph(prebuilt_fake).ainvoke(
        {"messages": [HumanMessage(content="Нужна ли эскалация?")]}
    )

    assert custom["messages"][-1].content == prebuilt["messages"][-1].content == answer
    assert custom_fake.received[0][0].content == agent.SYSTEM_PROMPT
    assert prebuilt_fake.received[0][0].content == agent.SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_unknown_tool_becomes_tool_message_instead_of_crashing():
    state = {
        "messages": [HumanMessage(content="Проверь баланс"), _tool_call("get_user_balance", {"user_id": "7"})],
        "iteration_count": 1,
        "tool_results": [],
    }

    update = await agent.execute_tool(state)

    assert isinstance(update["messages"][0], ToolMessage)
    assert "неизвестный tool get_user_balance" in update["messages"][0].content
    assert update["tool_results"] == [
        {"tool_name": "get_user_balance", "tool_call_id": "call-1", "status": "error", "result": "Ошибка: неизвестный tool get_user_balance."}
    ]


@pytest.mark.asyncio
async def test_sixth_model_turn_forces_explicit_finish_after_useless_tool(monkeypatch):
    # A deliberately unknown tool simulates a model that keeps making useless
    # attempts; the graph must finish rather than loop indefinitely.
    fake = _FakeBoundModel([_tool_call("useless_tool", {}, f"call-{number}") for number in range(1, 7)])
    monkeypatch.setattr(agent, "bound_model", fake)

    result = await agent.custom_graph.ainvoke(
        {"messages": [HumanMessage(content="Продолжай вызывать время")], "iteration_count": 0, "tool_results": []},
        config={"recursion_limit": 20},
    )

    assert result["iteration_count"] == agent.MAX_ITERATIONS
    assert len(result["tool_results"]) == agent.MAX_ITERATIONS - 1
    assert {item["status"] for item in result["tool_results"]} == {"error"}
    assert result["messages"][-1].content == agent.MAX_ITERATIONS_MESSAGE


def test_router_is_pure_literal_decision_and_has_priority_for_ceiling():
    has_tool = {"messages": [_tool_call()], "iteration_count": 1, "tool_results": []}
    final = {"messages": [AIMessage(content="Готово")], "iteration_count": 1, "tool_results": []}
    ceiling = {"messages": [_tool_call()], "iteration_count": agent.MAX_ITERATIONS, "tool_results": []}

    assert agent.route_after_model(has_tool) == "execute_tool"
    assert agent.route_after_model(final) == "force_finish"
    assert agent.route_after_model(ceiling) == "force_finish"
    assert ceiling["iteration_count"] == agent.MAX_ITERATIONS


@pytest.mark.asyncio
async def test_tool_records_and_message_state_have_a_serializable_representation():
    state = {
        "messages": [HumanMessage(content="Неизвестный инструмент"), _tool_call("not_allowed", {}, "call-json")],
        "iteration_count": 1,
        "tool_results": [],
    }
    update = await agent.execute_tool(state)
    serializable = {
        "messages": messages_to_dict(state["messages"] + update["messages"]),
        "iteration_count": state["iteration_count"],
        "tool_results": update["tool_results"],
    }

    restored = json.loads(json.dumps(serializable, ensure_ascii=False))
    assert restored["tool_results"][0]["status"] == "error"
    assert "get_user_balance" not in agent.TOOL_BY_NAME


def test_graph_topology_and_shared_tool_contract_are_exposed():
    custom_mermaid = agent.custom_graph.get_graph().draw_mermaid()
    prebuilt_mermaid = agent.prebuilt_graph.get_graph().draw_mermaid()

    assert "call_model" in custom_mermaid
    assert "execute_tool" in custom_mermaid
    assert "force_finish" in custom_mermaid
    assert "model" in prebuilt_mermaid and "tools" in prebuilt_mermaid
    assert {tool.name for tool in agent.TOOLS} == {"search_knowledge_base", "get_current_time", "send_telegram_message"}
    assert "ReAct-агент поддержки MIDAS" in agent.SYSTEM_PROMPT


def test_benchmark_uses_exact_six_two_cases_and_requires_three_repeats():
    assert bench_agents.CASES is CASES
    assert bench_agents.DEFAULT_REPEATS == 3
    with pytest.raises(ValueError, match="at least 3"):
        asyncio.run(bench_agents.run_benchmark(repeats=2))
