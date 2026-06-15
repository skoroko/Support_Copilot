from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from support_copilot.infrastructure.async_llm import AsyncLLMClient

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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat/stream")
async def chat_stream(payload: ChatStreamRequest, request: Request) -> Any:
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
            logger.exception("sse.stream_failed")
            yield {
                "event": "error",
                "data": json.dumps({"type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False),
            }

    if EventSourceResponse is not None:
        return EventSourceResponse(events(), ping=15)
    return StreamingResponse(_encode_sse(events()), media_type="text/event-stream")


async def _encode_sse(events: AsyncIterator[dict[str, str]]) -> AsyncIterator[str]:
    async for event in events:
        name = event.get("event", "message")
        data = event.get("data", "")
        yield f"event: {name}\ndata: {data}\n\n"
