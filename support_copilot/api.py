from __future__ import annotations

import json
import logging
import os
import secrets
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from support_copilot.infrastructure.async_llm import AsyncLLMClient
from app.core.auth import InMemoryRequestBudget

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:  # pragma: no cover - dependency is documented in requirements
    EventSourceResponse = None


logger = logging.getLogger(__name__)
app = FastAPI(title="Support Copilot API", version="0.3.3")


class ChatStreamRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    temperature: float = 0.2
    max_tokens: int = 500


def get_async_client() -> AsyncLLMClient:
    client = getattr(app.state, "async_llm_client", None)
    if client is None:
        client = AsyncLLMClient.from_env(enable_cache=True, logger=logger)
        app.state.async_llm_client = client
    return client


async def require_standalone_principal(
    request: Request,
    x_internal_token: Annotated[str | None, Header(alias="X-Internal-Token")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
) -> AsyncIterator[str]:
    expected = os.environ.get("INTERNAL_TOKEN", "").strip()
    if len(expected) < 32 or not x_internal_token or not secrets.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=401, detail="Valid service authentication is required")
    principal = (x_user_id or "").strip()
    if not principal or len(principal) > 200:
        raise HTTPException(status_code=401, detail="Authenticated principal is required")
    budget = getattr(app.state, "request_budget", None)
    if budget is None:
        budget = InMemoryRequestBudget(requests_per_minute=60, max_concurrency=4)
        app.state.request_budget = budget
    await budget.acquire(principal)
    try:
        yield principal
    finally:
        await budget.release(principal)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat/stream")
async def chat_stream(
    payload: ChatStreamRequest,
    request: Request,
    _principal: Annotated[str, Depends(require_standalone_principal)],
) -> Any:
    client = get_async_client()

    async def events() -> AsyncIterator[dict[str, str]]:
        start = time.perf_counter()
        first_token_at: float | None = None
        try:
            async for delta in client.stream_chat(
                payload.prompt,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
            ):
                if await request.is_disconnected():
                    logger.info("sse.client_disconnected prompt_chars=%d", len(payload.prompt))
                    return
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    yield {
                        "event": "meta",
                        "data": json.dumps(
                            {"ttft_ms": round((first_token_at - start) * 1000, 2)},
                            ensure_ascii=False,
                        ),
                    }
                yield {"event": "delta", "data": delta}
            total_ms = (time.perf_counter() - start) * 1000
            yield {"event": "done", "data": json.dumps({"total_ms": round(total_ms, 2)})}
        except Exception as exc:  # noqa: BLE001 - SSE boundary returns structured error
            logger.warning("sse.stream_failed", extra={"error_type": type(exc).__name__})
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "type": "error",
                        "code": "chat_stream_failed",
                        "message": "Не удалось завершить ответ. Повторите запрос позже.",
                    },
                    ensure_ascii=False,
                ),
            }

    if EventSourceResponse is not None:
        return EventSourceResponse(events(), ping=15)
    return StreamingResponse(_encode_sse(events()), media_type="text/event-stream")


async def _encode_sse(events: AsyncIterator[dict[str, str]]) -> AsyncIterator[str]:
    async for event in events:
        name = event.get("event", "message")
        data = event.get("data", "")
        yield f"event: {name}\ndata: {data}\n\n"
