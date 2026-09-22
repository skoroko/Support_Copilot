"""LangGraph implementation of the bounded Support Copilot ReAct agent.

The module intentionally keeps the graph state small and checkpoint-friendly:
only LangChain messages, integers and JSON-like dictionaries cross node
boundaries.  A provider client or an API key is never stored in ``AgentState``.
``custom_graph`` exposes the explicit learning graph, while ``prebuilt_graph``
uses LangChain's equivalent ``create_agent`` helper with the same model, tools
and system prompt.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.services.agent_naive_tools import (
    get_current_time as _get_current_time,
    search_knowledge_base as _search_knowledge_base,
    send_telegram_message as _send_telegram_message,
)
from app.services.agent_react import SYSTEM_PROMPT
from support_copilot.config import load_settings


DEFAULT_MODEL = "gpt-5.4-mini"
MAX_ITERATIONS = 6
MAX_ITERATIONS_MESSAGE = (
    "Превышен лимит из 6 итераций. Для безопасного ответа нужна проверка инженером поддержки."
)


class AgentState(TypedDict):
    """Serializable graph state; reducers are explicit at the contract edge."""

    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    tool_results: Annotated[list[dict[str, Any]], operator.add]


@tool
def search_knowledge_base(query: str) -> str:
    """Find one reliable MIDAS support answer in the local knowledge base.

    Use this only for a short factual support query.  It returns the best local
    fragment with its source or an explicit request for engineer verification;
    it never invents product capabilities, prices or regulations.
    """

    return _search_knowledge_base(query)


@tool
def get_current_time(timezone: str = "Europe/Moscow") -> str:
    """Return local ISO-8601 time for an IANA timezone without a network call.

    The input is an IANA timezone such as ``Europe/Moscow``.  The result is a
    time value only and must not be treated as evidence about licences,
    schedules or service policies.
    """

    return _get_current_time(timezone)


@tool
def send_telegram_message(chat_id: str, text: str) -> str:
    """Safely simulate a Telegram notification after explicit confirmation.

    Both ``chat_id`` and ``text`` must have been explicitly confirmed by the
    user.  This is a local educational stub, so its result does not claim that
    a message reached a real Telegram recipient.
    """

    return _send_telegram_message(chat_id, text)


TOOLS: tuple[BaseTool, ...] = (
    search_knowledge_base,
    get_current_time,
    send_telegram_message,
)
TOOL_BY_NAME: dict[str, BaseTool] = {tool_.name: tool_ for tool_ in TOOLS}


def create_chat_model() -> ChatOpenAI:
    """Create an OpenAI-compatible model without requiring a key at import time.

    ``not-configured`` is deliberately a non-secret placeholder.  It permits
    offline graph inspection and mock-based tests; a live invocation still
    fails clearly until ``OPENAI_API_KEY`` is configured.
    """

    settings = load_settings()
    return ChatOpenAI(
        model=settings.openai_model or DEFAULT_MODEL,
        api_key=settings.openai_api_key or "not-configured",
        base_url=settings.openai_base_url,
        timeout=settings.llm_timeout_sec,
        max_retries=0,
        temperature=0,
    )


model = create_chat_model()
bound_model = model.bind_tools(TOOLS)


async def call_model(state: AgentState) -> dict[str, Any]:
    """Ask the tool-bound model for one next message and count that model turn."""

    messages = list(state["messages"])
    # Unlike ``create_agent(system_prompt=...)``, a raw ChatOpenAI binding does
    # not own a system prompt.  Add it at invocation time (not to state) so the
    # custom and prebuilt variants follow exactly the same policy without
    # duplicating a SystemMessage on every tool loop.
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    response = await bound_model.ainvoke(messages)
    return {
        "messages": [response],
        # No reducer: each node replaces this scalar with the next count.
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


async def execute_tool(state: AgentState) -> dict[str, Any]:
    """Execute every requested allowlisted tool and turn failures into messages.

    Unknown names and malformed arguments remain visible to the model as
    ``ToolMessage`` values instead of causing a graph crash.
    """

    last_message = state["messages"][-1]
    tool_calls = list(getattr(last_message, "tool_calls", []) or [])
    messages: list[ToolMessage] = []
    results: list[dict[str, Any]] = []

    for raw_call in tool_calls:
        name = str(raw_call.get("name", ""))
        call_id = str(raw_call.get("id", "missing-tool-call-id"))
        raw_args = raw_call.get("args", {})
        args = raw_args if isinstance(raw_args, dict) else {}
        handler = TOOL_BY_NAME.get(name)

        if handler is None:
            content = f"Ошибка: неизвестный tool {name or '<empty>'}."
            status = "error"
        elif not isinstance(raw_args, dict):
            content = f"Ошибка tool {name}: аргументы должны быть JSON object."
            status = "error"
        else:
            try:
                content = str(await handler.ainvoke(args))
                status = "ok"
            except Exception as exc:  # The model receives a bounded domain error.
                content = f"Ошибка tool {name}: {type(exc).__name__}: {exc}"
                status = "error"

        messages.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
        results.append(
            {
                "tool_name": name,
                "tool_call_id": call_id,
                "status": status,
                "result": content,
            }
        )

    return {"messages": messages, "tool_results": results}


async def force_finish(state: AgentState) -> dict[str, Any]:
    """End explicitly at the iteration ceiling or preserve the final AI answer."""

    last_message = state["messages"][-1] if state.get("messages") else None
    if state.get("iteration_count", 0) >= MAX_ITERATIONS and getattr(last_message, "tool_calls", None):
        return {"messages": [AIMessage(content=MAX_ITERATIONS_MESSAGE)]}

    # The final model message has already been appended by ``call_model``.  An
    # empty update deliberately avoids duplicating it while making the terminal
    # node explicit in the graph rather than routing straight to END.
    return {"messages": []}


def route_after_model(state: AgentState) -> Literal["execute_tool", "force_finish"]:
    """Pure, deterministic router for the custom graph's only branch."""

    last_message = state["messages"][-1] if state.get("messages") else None
    if not getattr(last_message, "tool_calls", None):
        return "force_finish"
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "force_finish"
    if getattr(last_message, "tool_calls", None):
        return "execute_tool"
    return "force_finish"


def build_custom_graph():
    """Compile the explicit START → model → tools/finish StateGraph."""

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("force_finish", force_finish)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {"execute_tool": "execute_tool", "force_finish": "force_finish"},
    )
    builder.add_edge("execute_tool", "call_model")
    builder.add_edge("force_finish", END)
    return builder.compile()


def build_prebuilt_graph(chat_model: Any = model):
    """Compile LangChain's prebuilt agent from the same model, tools and prompt."""

    return create_agent(chat_model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


custom_graph = build_custom_graph()
prebuilt_graph = build_prebuilt_graph()


__all__ = [
    "AgentState",
    "MAX_ITERATIONS",
    "MAX_ITERATIONS_MESSAGE",
    "SYSTEM_PROMPT",
    "TOOLS",
    "bound_model",
    "call_model",
    "custom_graph",
    "execute_tool",
    "force_finish",
    "model",
    "prebuilt_graph",
    "route_after_model",
]
