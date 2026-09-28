"""SSE transport for the persistent, human-approved LangGraph agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated, Any, Literal
from weakref import WeakValueDictionary

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.auth import InMemoryRequestBudget, authenticate_principal, request_budget
from app.services.agent_persistent import Command

router = APIRouter(prefix="/agent", tags=["agent"])
logger = logging.getLogger(__name__)
_CLEANUP_WAIT_SECONDS = 1.0


class AgentUserMessage(BaseModel):
    """Only an ordinary user message may cross the public graph boundary."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user"] = "user"
    content: str = Field(min_length=1, max_length=8_000)


class AgentTurnInput(BaseModel):
    """Bound one turn before it reaches the persistent state reducer."""

    model_config = ConfigDict(extra="forbid")

    messages: list[AgentUserMessage] = Field(min_length=1, max_length=4)


class AgentStreamRequest(BaseModel):
    """One authenticated graph turn or one approval resume command."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._:-]+$")
    input: AgentTurnInput | None = None
    resume: bool | None = None
    user_role: Literal["read-only", "write-with-approve", "full"] = "write-with-approve"

    @model_validator(mode="after")
    def require_exactly_one_action(self) -> AgentStreamRequest:
        if (self.input is None) == (self.resume is None):
            raise ValueError("Provide exactly one of input or resume")
        return self


def _json_safe(value: Any) -> Any:
    """Convert LangGraph/LangChain event values to conservative JSON-safe data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump(mode="json"))
    if hasattr(value, "value"):
        return {"value": _json_safe(value.value)}
    if hasattr(value, "content"):
        return {"content": _json_safe(value.content)}
    return str(value)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(_json_safe(payload), ensure_ascii=False, separators=(',', ':'))}\n\n"


def _snapshot_interrupts(snapshot: Any) -> list[Any]:
    interrupts: list[Any] = []
    for task in getattr(snapshot, "tasks", ()) or ():
        interrupts.extend(getattr(task, "interrupts", ()) or ())
    return interrupts


def _authorize_role(settings: Any, *, user_role: str, admin_token: str | None) -> None:
    """Keep the full agent role restricted to the administrative caller."""

    if user_role == "full":
        expected_admin = settings.admin_token.get_secret_value()
        if not admin_token or not secrets.compare_digest(admin_token, expected_admin):
            raise HTTPException(status_code=403, detail="The full role requires X-Admin-Token")


class _BudgetedStreamingResponse(StreamingResponse):
    """Hold the shared principal slot until ASGI finishes or disconnects."""

    def __init__(
        self,
        *args: Any,
        budget: InMemoryRequestBudget,
        principal: str,
        drive: Callable[[], Awaitable[None]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._budget = budget
        self._principal = principal
        self._drive = drive
        self._driver: asyncio.Task[None] | None = None

    async def _finish(self) -> None:
        try:
            await self.body_iterator.aclose()
        finally:
            try:
                if self._driver is not None:
                    try:
                        await self._driver
                    except asyncio.CancelledError:
                        if not self._driver.cancelled():
                            raise
            finally:
                await self._budget.release(self._principal)

    @staticmethod
    def _report_background_failure(task: asyncio.Task[None]) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error("Agent stream cleanup failed after disconnect")

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        # Starlette's ASGI 2.3 disconnect task group cancels the body iterator
        # repeatedly. Keep the graph outside that cancel scope so its async
        # finalizers receive only one controlled cancellation.
        self._driver = asyncio.create_task(self._drive())
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._driver.cancel()
            finish = asyncio.create_task(self._finish())
            finish.add_done_callback(self._report_background_failure)
            try:
                await asyncio.wait_for(asyncio.shield(finish), timeout=_CLEANUP_WAIT_SECONDS)
            except TimeoutError:
                # The cleanup task retains the budget until the graph exits.
                pass


def _private_thread_id(user_id: str, client_thread_id: str) -> str:
    material = f"{user_id}\0{client_thread_id}".encode()
    return f"agent-{hashlib.sha256(material).hexdigest()}"


@router.post("/stream", response_class=StreamingResponse)
async def stream_agent(
    body: AgentStreamRequest,
    request: Request,
    x_internal_token: Annotated[str | None, Header(alias="X-Internal-Token")] = None,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
    x_user_interface: Annotated[str | None, Header(alias="X-User-Interface")] = None,
) -> StreamingResponse:
    """Stream graph updates/messages and surface an approval pause as SSE JSON."""

    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=503, detail="Application settings are not initialized")
    principal = authenticate_principal(
        settings,
        x_internal_token=x_internal_token,
        x_user_id=x_user_id,
        x_user_interface=x_user_interface,
    )
    _authorize_role(settings, user_role=body.user_role, admin_token=x_admin_token)
    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="Persistent agent is not initialized")

    thread_id = _private_thread_id(principal.user_id, body.thread_id)
    turn_locks = getattr(request.app.state, "agent_turn_locks", None)
    if turn_locks is None:
        turn_locks = WeakValueDictionary()
        request.app.state.agent_turn_locks = turn_locks
    turn_lock = turn_locks.get(thread_id)
    if turn_lock is None:
        turn_lock = asyncio.Lock()
        turn_locks[thread_id] = turn_lock

    graph_input: Any = (
        Command(resume=body.resume)
        if body.resume is not None
        else body.input.model_dump()
    )
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_role": body.user_role,
        }
    }

    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=1)

    async def drive() -> None:
        interrupt_emitted = False
        cleanup_tasks: set[asyncio.Task[Any]] = set()

        def remember_cleanup(exc: asyncio.CancelledError) -> None:
            cleanup_tasks.update(arg for arg in exc.args if isinstance(arg, asyncio.Task))

        try:
            async with turn_lock:
                graph_stream = graph.astream(
                    graph_input,
                    config=config,
                    stream_mode=["updates", "messages"],
                )
                try:
                    async for stream_type, payload in graph_stream:
                        event = {"type": stream_type, "payload": _json_safe(payload)}
                        if stream_type == "updates" and isinstance(payload, dict) and "__interrupt__" in payload:
                            interrupt_emitted = True
                            event["interrupt"] = _json_safe(payload["__interrupt__"])
                        await queue.put(_sse(event))
                except asyncio.CancelledError as exc:
                    remember_cleanup(exc)
                    raise
                finally:
                    try:
                        await graph_stream.aclose()
                    except asyncio.CancelledError as exc:
                        remember_cleanup(exc)
                        raise
                    finally:
                        if cleanup_tasks:
                            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                            for result in results:
                                if isinstance(result, BaseException):
                                    raise result

                # Some graph versions emit ``__interrupt__`` only in the final
                # state, not in the updates stream. Emit one portable pause event.
                if not interrupt_emitted:
                    snapshot = await graph.aget_state(config)
                    interrupts = _snapshot_interrupts(snapshot)
                    if interrupts:
                        await queue.put(_sse({"type": "interrupt", "payload": _json_safe(interrupts)}))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - send only a fixed safe error after headers
            await queue.put(_sse(
                {
                    "type": "error",
                    "error": {
                        "code": "agent_stream_failed",
                        "message": "Agent stream failed; retry or escalate to support.",
                    },
                }
            ))
        else:
            await queue.put(_sse({"type": "done"}))
        await queue.put(None)

    async def events() -> AsyncIterator[str]:
        while (event := await queue.get()) is not None:
            yield event

    budget = request_budget(request, settings)
    await budget.acquire(principal.user_id)
    return _BudgetedStreamingResponse(
        events(),
        budget=budget,
        principal=principal.user_id,
        drive=drive,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["AgentStreamRequest", "AgentTurnInput", "AgentUserMessage", "router"]
