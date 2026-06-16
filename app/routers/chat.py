from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.deps.providers import LLMServiceDep
from app.schemas.chat import ChatRequest, ChatResponse


RESPONSES = {
    200: {"description": "Successful response"},
    422: {"description": "Validation error"},
    429: {"description": "LLM provider rate limit"},
    502: {"description": "LLM provider/auth error"},
    504: {"description": "LLM provider timeout"},
}


router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Complete one chat request",
    responses=RESPONSES,
)
async def complete_chat(req: ChatRequest, service: LLMServiceDep) -> ChatResponse:
    return await service.complete(req)


@router.post(
    "/chat/stream",
    summary="Stream chat completion as Server-Sent Events",
    responses=RESPONSES,
)
async def stream_chat(req: ChatRequest, service: LLMServiceDep) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        async for delta in service.stream(req.model_copy(update={"stream": True})):
            if delta.usage is not None:
                yield f"data: {json.dumps({'usage': delta.usage.model_dump()}, ensure_ascii=False)}\n\n"
            elif delta.content is not None:
                yield f"data: {delta.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
