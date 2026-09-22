"""SSE transport for the persistent, human-approved LangGraph agent."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.agent_persistent import Command


router = APIRouter(prefix="/agent", tags=["agent"])


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
    def require_exactly_one_action(self) -> "AgentStreamRequest":
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


def _authorize(
    request: Request,
    *,
    user_role: str,
    internal_token: str | None,
    admin_token: str | None,
    user_id: str | None,
) -> str:
    """Authenticate the internal caller and derive a private thread namespace."""

    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=503, detail="Application settings are not initialized")
    expected_internal = settings.internal_token.get_secret_value()
    if not internal_token or not secrets.compare_digest(internal_token, expected_internal):
        raise HTTPException(status_code=401, detail="Valid X-Internal-Token is required")
    if not user_id or len(user_id) > 200:
        raise HTTPException(status_code=401, detail="Valid X-User-ID is required")
    if user_role == "full":
        expected_admin = settings.admin_token.get_secret_value()
        if not admin_token or not secrets.compare_digest(admin_token, expected_admin):
            raise HTTPException(status_code=403, detail="The full role requires X-Admin-Token")
    return user_id


def _private_thread_id(user_id: str, client_thread_id: str) -> str:
    material = f"{user_id}\0{client_thread_id}".encode("utf-8")
    return f"agent-{hashlib.sha256(material).hexdigest()}"


@router.post("/stream", response_class=StreamingResponse)
async def stream_agent(
    body: AgentStreamRequest,
    request: Request,
    x_internal_token: Annotated[str | None, Header(alias="X-Internal-Token")] = None,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
) -> StreamingResponse:
    """Stream graph updates/messages and surface an approval pause as SSE JSON."""

    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="Persistent agent is not initialized")

    user_id = _authorize(
        request,
        user_role=body.user_role,
        internal_token=x_internal_token,
        admin_token=x_admin_token,
        user_id=x_user_id,
    )
    graph_input: Any = (
        Command(resume=body.resume)
        if body.resume is not None
        else body.input.model_dump()
    )
    config = {
        "configurable": {
            "thread_id": _private_thread_id(user_id, body.thread_id),
            "user_role": body.user_role,
        }
    }

    async def events() -> AsyncIterator[str]:
        interrupt_emitted = False
        try:
            async for stream_type, payload in graph.astream(
                graph_input,
                config=config,
                stream_mode=["updates", "messages"],
            ):
                event = {"type": stream_type, "payload": _json_safe(payload)}
                if stream_type == "updates" and isinstance(payload, dict) and "__interrupt__" in payload:
                    interrupt_emitted = True
                    event["interrupt"] = _json_safe(payload["__interrupt__"])
                yield _sse(event)

            # Some graph versions emit ``__interrupt__`` only in the final
            # state, not in the updates stream.  Emit one portable pause event.
            if not interrupt_emitted:
                snapshot = await graph.aget_state(config)
                interrupts = _snapshot_interrupts(snapshot)
                if interrupts:
                    yield _sse({"type": "interrupt", "payload": _json_safe(interrupts)})
        except Exception:
            yield _sse(
                {
                    "type": "error",
                    "error": {
                        "code": "agent_stream_failed",
                        "message": "Agent stream failed; retry or escalate to support.",
                    },
                }
            )
            return
        yield _sse({"type": "done"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["AgentStreamRequest", "AgentTurnInput", "AgentUserMessage", "router"]
