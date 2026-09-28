"""Persistent LangGraph agent with human approval for Telegram delivery.

Homework 6.3 deliberately keeps its in-memory graph in ``agent_graph.py``.
This module is a separate, checkpoint-backed increment: a request is durable
when a stable ``thread_id`` is supplied, while the one dangerous domain action
is paused with ``interrupt()`` before the side effect can happen.

Only JSON-like data and LangChain messages are placed in ``PersistentState``.
Provider clients, connections and callable dependencies live in node closures,
not in checkpointed state.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from threading import Thread
from typing import Annotated, Any, Literal, TypedDict

from typing_extensions import NotRequired

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from app.core.config import PROJECT_ROOT, get_settings
from app.services.agent_naive_tools import get_current_time as _get_current_time
from app.services.agent_naive_tools import search_knowledge_base as _search_knowledge_base
from app.services.notifier import notify_user
from app.services.agent_react import SYSTEM_PROMPT


CHECKPOINTER_BACKENDS = frozenset({"memory", "sqlite", "postgres"})
DEFAULT_CHECKPOINTER_BACKEND = "sqlite"
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "var" / "agent_checkpoints.sqlite"
DANGEROUS_TOOL_NAME = "send_telegram_message"
APPROVAL_NODE = "confirm_and_execute_telegram_message"
PREPARE_NODE = "prepare_telegram_message"
MAX_ITERATIONS = 6
MAX_ITERATIONS_MESSAGE = (
    "Превышен лимит из 6 итераций. Для безопасного ответа нужна проверка инженером поддержки."
)
PERSISTENT_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "tool является только локальной заглушкой.",
    "после подготовки действие обязательно проходит persistent human approval и только затем вызывает внутренний Telegram API.",
)

UserRole = Literal["read-only", "write-with-approve", "full"]
SendMessage = Callable[[str, str, str], Awaitable[str] | str]


class PersistentState(TypedDict):
    """The complete serializable contract persisted by a LangGraph saver."""

    messages: Annotated[list[AnyMessage], add_messages]
    tool_results: Annotated[list[dict[str, Any]], list.__add__]
    telegram_draft: NotRequired[dict[str, str] | None]
    sent: NotRequired[bool]
    iteration_count: NotRequired[int]


@tool
def search_knowledge_base(query: str) -> str:
    """Search the local MIDAS knowledge base for a factual support answer."""

    return _search_knowledge_base(query)


@tool
def get_current_time(timezone: str = "Europe/Moscow") -> str:
    """Return local ISO-8601 time for an IANA timezone without network access."""

    return _get_current_time(timezone)


@tool
def send_telegram_message(chat_id: str, text: str) -> str:
    """Request an explicitly approved Telegram delivery to the exact chat and text.

    The graph never invokes this tool directly.  It first creates a durable
    preview and then pauses for approval in a dedicated node.  The real delivery
    adapter is injected into :func:`build_agent`, which keeps tests offline.
    """

    # This body is only a schema/documentation fallback.  Direct execution is
    # intentionally prohibited by the graph's router.
    return f"Delivery request prepared for {chat_id}: {text}"


TOOLS: tuple[BaseTool, ...] = (search_knowledge_base, get_current_time, send_telegram_message)
SAFE_TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "search_knowledge_base": _search_knowledge_base,
    "get_current_time": _get_current_time,
}


def _latest_tool_calls(state: PersistentState) -> list[dict[str, Any]]:
    """Return every call from the latest assistant turn in provider order."""

    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            calls = getattr(message, "tool_calls", None) or []
            return [dict(call) for call in calls if isinstance(call, dict)]
    return []


def _first_tool_call(state: PersistentState) -> dict[str, Any] | None:
    """Return the first latest call for compatibility with single-call paths."""

    calls = _latest_tool_calls(state)
    return calls[0] if calls else None


def _tool_message(call: dict[str, Any], content: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        tool_call_id=str(call.get("id", "missing-tool-call-id")),
        name=str(call.get("name", "unknown")),
    )


def _tool_result(call: dict[str, Any], *, status: str, result: str) -> dict[str, str]:
    return {
        "tool_name": str(call.get("name", "unknown")),
        "tool_call_id": str(call.get("id", "missing-tool-call-id")),
        "status": status,
        "result": result,
    }


def _role(config: RunnableConfig | None) -> UserRole:
    value = ((config or {}).get("configurable") or {}).get("user_role", "write-with-approve")
    if value in {"read-only", "write-with-approve", "full"}:
        return value
    return "write-with-approve"


async def _default_send_message(chat_id: str, text: str, idempotency_key: str) -> str:
    """Deliver through the existing authenticated internal bot HTTP endpoint."""

    try:
        numeric_chat_id = int(chat_id)
    except ValueError as exc:
        raise ValueError("Telegram chat_id must be an integer") from exc
    await notify_user(
        numeric_chat_id,
        text,
        settings=get_settings(),
        idempotency_key=idempotency_key,
    )
    return f"Сообщение передано Telegram-боту для чата {chat_id}."


def _resolve_effect(outcome: Awaitable[str] | str) -> str:
    """Resolve an async adapter from the synchronous interrupt node safely.

    LangGraph 1.x needs ``interrupt()`` to execute in a synchronous node on
    the project's Python 3.10 runtime because contextvars cannot be supplied to
    async node tasks there.  The adapter itself is still async-friendly (and
    is an ``AsyncMock`` in tests), so its coroutine runs in a short dedicated
    thread only after approval.  Python 3.11+ can keep this compatibility path
    until the project raises its minimum Python version.
    """

    if not inspect.isawaitable(outcome):
        return str(outcome)
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = asyncio.run(outcome)
        except BaseException as exc:  # re-raise after joining the worker
            box["error"] = exc

    worker = Thread(target=run, name="agent-telegram-effect", daemon=False)
    worker.start()
    worker.join()
    if "error" in box:
        raise box["error"]
    return str(box["value"])


def _interrupt_with_config(config: RunnableConfig, payload: dict[str, Any]) -> Any:
    """Call the public interrupt API on both Python 3.10 and newer runtimes.

    LangGraph's normal async runner propagates its contextvar on Python 3.11+.
    The local course environment still runs Python 3.10, where a synchronous
    node is dispatched through an executor without that context.  Re-install
    the already supplied node configuration only for that compatibility case;
    ``interrupt`` itself and ``Command(resume=...)`` remain the public HIL API.
    """

    try:
        return interrupt(payload)
    except RuntimeError as exc:
        if "outside of a runnable context" not in str(exc):
            raise
        from langchain_core.runnables.config import set_config_context

        with set_config_context(config) as context:
            return context.run(interrupt, payload)


def _default_model() -> Any:
    """Lazily create a bound model so graph imports stay safe for offline tests."""

    settings = get_settings()
    return ChatOpenAI(
        model=settings.default_model,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        timeout=settings.request_timeout,
        max_retries=0,
        temperature=0,
    ).bind_tools(TOOLS)


def build_agent(
    checkpointer: Any,
    *,
    model: Any | None = None,
    send_message: SendMessage = _default_send_message,
):
    """Compile the persistent ReAct-like graph around an injected checkpointer.

    ``model`` and ``send_message`` are dependencies rather than state.  This is
    intentional: checkpoints remain serializable and offline tests can replace
    the side effect with ``AsyncMock``.
    """

    resolved_model = model

    def begin_turn(state: PersistentState) -> dict[str, int]:
        # START is entered for a fresh input, but not for Command(resume=...).
        # Keep the bound on one user turn instead of the lifetime of its thread.
        del state
        return {"iteration_count": 0}

    async def call_model(state: PersistentState, config: RunnableConfig) -> dict[str, Any]:
        # ``config`` is accepted as a LangGraph node argument so its
        # configurable contract remains available to model adapters later.
        del config
        nonlocal resolved_model
        if resolved_model is None:
            resolved_model = _default_model()
        messages = list(state.get("messages", []))
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=PERSISTENT_SYSTEM_PROMPT))
        response = await resolved_model.ainvoke(messages)
        return {
            "messages": [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    def route_after_model(state: PersistentState) -> Literal["execute_safe_tool", "prepare_telegram_message", "finish"]:
        calls = _latest_tool_calls(state)
        if not calls:
            return "finish"
        if state.get("iteration_count", 0) >= MAX_ITERATIONS:
            return "finish"
        if any(call.get("name") == DANGEROUS_TOOL_NAME for call in calls):
            return PREPARE_NODE
        return "execute_safe_tool"

    def execute_one_safe_call(call: dict[str, Any]) -> tuple[ToolMessage, dict[str, str]]:
        name = str(call.get("name", ""))
        args = call.get("args")
        handler = SAFE_TOOL_HANDLERS.get(name)
        if handler is None:
            content = f"Ошибка: неизвестный или требующий подтверждения tool {name or '<empty>'}."
            status = "error"
        elif not isinstance(args, dict):
            content = f"Ошибка tool {name}: аргументы должны быть JSON object."
            status = "error"
        else:
            try:
                content = str(handler(**args))
                status = "ok"
            except Exception as exc:  # expose a bounded tool error to the model
                content = f"Ошибка tool {name}: {type(exc).__name__}: {exc}"
                status = "error"
        return _tool_message(call, content), _tool_result(call, status=status, result=content)

    async def execute_safe_tool(state: PersistentState) -> dict[str, Any]:
        calls = _latest_tool_calls(state)
        if not calls:
            return {"messages": [], "tool_results": []}
        outcomes = [execute_one_safe_call(call) for call in calls]
        return {
            "messages": [message for message, _result in outcomes],
            "tool_results": [result for _message, result in outcomes],
        }

    async def prepare_telegram_message(
        state: PersistentState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Create a deterministic, side-effect-free delivery preview.

        This node is safe to replay: it validates and stores exactly the same
        ``chat_id``/``text`` from the model tool call and never contacts Telegram.
        """

        calls = _latest_tool_calls(state)
        dangerous_calls = [call for call in calls if call.get("name") == DANGEROUS_TOOL_NAME]
        call = dangerous_calls[0] if dangerous_calls else None
        messages: list[ToolMessage] = []
        results: list[dict[str, str]] = []
        for other in calls:
            if other is call:
                continue
            if other.get("name") == DANGEROUS_TOOL_NAME:
                content = "Отправка не выполнена: за один model turn разрешено одно внешнее действие."
                messages.append(_tool_message(other, content))
                results.append(_tool_result(other, status="rejected", result=content))
            else:
                message, result = execute_one_safe_call(other)
                messages.append(message)
                results.append(result)

        args = call.get("args") if call else None
        if not isinstance(args, dict) or not isinstance(args.get("chat_id"), str) or not isinstance(args.get("text"), str):
            content = "Ошибка: для Telegram нужны строковые chat_id и text."
            if call is not None:
                messages.append(_tool_message(call, content))
                results.append(_tool_result(call, status="error", result=content))
            return {
                "telegram_draft": None,
                "sent": False,
                "messages": messages,
                "tool_results": results,
            }
        configurable = config.get("configurable") or {}
        thread_id = str(configurable.get("thread_id", "missing-thread"))
        tool_call_id = str(call.get("id", "missing-tool-call-id"))
        key_material = f"{thread_id}\0{tool_call_id}".encode("utf-8")
        draft = {
            "chat_id": args["chat_id"],
            "text": args["text"],
            "tool_call_id": tool_call_id,
            "idempotency_key": hashlib.sha256(key_material).hexdigest(),
        }
        return {
            "telegram_draft": draft,
            "sent": False,
            "messages": messages,
            "tool_results": results,
        }

    def confirm_and_execute_telegram_message(
        state: PersistentState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Pause for approval and execute the injected side effect only afterwards."""

        draft = state.get("telegram_draft")
        if not draft:
            return {"messages": [], "sent": False, "tool_results": []}
        call = {
            "name": DANGEROUS_TOOL_NAME,
            "id": draft["tool_call_id"],
        }

        role = _role(config)
        if role == "read-only":
            content = "Отправка не выполнена: роль read-only не имеет права отправлять сообщения."
            return {
                "messages": [_tool_message(call, content)],
                "sent": False,
                "tool_results": [_tool_result(call, status="forbidden", result=content)],
            }

        # ``interrupt`` is before any external effect.  LangGraph restarts this
        # node on resume, therefore every preceding operation remains idempotent.
        approved = True if role == "full" else bool(
            _interrupt_with_config(
                config,
                {
                    "type": "approve_send_telegram_message",
                    "preview": dict(draft),
                    "user_role": role,
                },
            )
        )
        if not approved:
            content = "Отправка отменена пользователем; сообщение не было отправлено."
            return {
                "messages": [_tool_message(call, content)],
                "sent": False,
                "tool_results": [_tool_result(call, status="rejected", result=content)],
            }

        outcome = send_message(
            draft["chat_id"],
            draft["text"],
            draft["idempotency_key"],
        )
        content = _resolve_effect(outcome)
        return {
            "messages": [_tool_message(call, content)],
            "sent": True,
            "tool_results": [_tool_result(call, status="sent", result=content)],
        }

    def finish(state: PersistentState) -> dict[str, Any]:
        calls = _latest_tool_calls(state)
        if state.get("iteration_count", 0) >= MAX_ITERATIONS and calls:
            content = "Tool не выполнен: достигнут лимит из 6 model turns."
            messages: list[AnyMessage] = [_tool_message(call, content) for call in calls]
            results = [_tool_result(call, status="rejected", result=content) for call in calls]
            messages.append(AIMessage(content=MAX_ITERATIONS_MESSAGE))
            return {"messages": messages, "tool_results": results}
        return {}

    graph = StateGraph(PersistentState)
    graph.add_node("begin_turn", begin_turn)
    graph.add_node("call_model", call_model)
    graph.add_node("execute_safe_tool", execute_safe_tool)
    graph.add_node(PREPARE_NODE, prepare_telegram_message)
    graph.add_node(APPROVAL_NODE, confirm_and_execute_telegram_message)
    graph.add_node("finish", finish)
    graph.add_edge(START, "begin_turn")
    graph.add_edge("begin_turn", "call_model")
    graph.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "execute_safe_tool": "execute_safe_tool",
            PREPARE_NODE: PREPARE_NODE,
            "finish": "finish",
        },
    )
    graph.add_edge("execute_safe_tool", "call_model")
    graph.add_edge(PREPARE_NODE, APPROVAL_NODE)
    graph.add_edge(APPROVAL_NODE, "call_model")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


def _checkpointer_backend() -> str:
    backend = os.getenv("AGENT_CHECKPOINTER", DEFAULT_CHECKPOINTER_BACKEND).strip().lower()
    if backend not in CHECKPOINTER_BACKENDS:
        allowed = ", ".join(sorted(CHECKPOINTER_BACKENDS))
        raise ValueError(f"AGENT_CHECKPOINTER must be one of {allowed}, got {backend!r}")
    return backend


def _sqlite_path() -> Path:
    raw = os.getenv("AGENT_SQLITE_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_SQLITE_PATH


def _postgres_uri() -> str:
    """Build a psycopg-v3 URI for the existing FastAPI Postgres database."""

    explicit = os.getenv("AGENT_POSTGRES_URI", "").strip()
    if explicit:
        return explicit
    host = os.getenv("POSTGRES_HOST", "postgres").strip() or "postgres"
    port = os.getenv("POSTGRES_PORT", "5432").strip() or "5432"
    database = os.getenv("POSTGRES_DB", "support_copilot").strip() or "support_copilot"
    user = os.getenv("POSTGRES_USER", "support").strip() or "support"
    password = os.getenv("POSTGRES_PASSWORD", "support")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


@asynccontextmanager
async def agent_lifespan(app: Any) -> AsyncIterator[None]:
    """Open one checkpoint saver and call its ``setup`` once per app startup."""

    backend = _checkpointer_backend()
    stack = AsyncExitStack()
    try:
        if backend == "memory":
            saver: Any = InMemorySaver()
        elif backend == "sqlite":
            path = _sqlite_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(path)))
            await saver.setup()
        else:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            saver = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(_postgres_uri()))
            await saver.setup()

        app.state.agent_graph = build_agent(saver)
        app.state.agent_checkpointer = saver
        app.state.agent_checkpointer_backend = backend
        yield
    finally:
        await stack.aclose()


__all__ = [
    "APPROVAL_NODE",
    "CHECKPOINTER_BACKENDS",
    "Command",
    "DANGEROUS_TOOL_NAME",
    "DEFAULT_CHECKPOINTER_BACKEND",
    "MAX_ITERATIONS",
    "MAX_ITERATIONS_MESSAGE",
    "PERSISTENT_SYSTEM_PROMPT",
    "PersistentState",
    "PREPARE_NODE",
    "TOOLS",
    "agent_lifespan",
    "build_agent",
]
