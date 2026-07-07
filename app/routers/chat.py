from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.deps.providers import LLMServiceDep
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse
from app.services.security.input_validator import validate_input
from app.services.security.output_filter import build_security_system_prompt


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
async def complete_chat(req: ChatRequest, service: LLMServiceDep, request: Request) -> ChatResponse:
    if not _security_enabled(request):
        return await service.complete(req, security_enabled=False)

    _validate_chat_input(req)
    system_prompt, canary = _security_context(request)
    secured_req = _with_security_system_prompt(req, system_prompt)
    return await service.complete(secured_req, system_prompt=system_prompt, canary=canary)


@router.post(
    "/chat/stream",
    summary="Stream chat completion as Server-Sent Events",
    responses=RESPONSES,
)
async def stream_chat(req: ChatRequest, service: LLMServiceDep, request: Request) -> StreamingResponse:
    if not _security_enabled(request):
        async def insecure_events() -> AsyncIterator[str]:
            async for delta in service.stream(req.model_copy(update={"stream": True}), security_enabled=False):
                if delta.usage is not None:
                    yield f"data: {json.dumps({'usage': delta.usage.model_dump()}, ensure_ascii=False)}\n\n"
                elif delta.content is not None:
                    yield f"data: {delta.content}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(insecure_events(), media_type="text/event-stream")

    _validate_chat_input(req)
    system_prompt, canary = _security_context(request)
    secured_req = _with_security_system_prompt(req.model_copy(update={"stream": True}), system_prompt)

    async def events() -> AsyncIterator[str]:
        async for delta in service.stream(secured_req, system_prompt=system_prompt, canary=canary):
            if delta.usage is not None:
                yield f"data: {json.dumps({'usage': delta.usage.model_dump()}, ensure_ascii=False)}\n\n"
            elif delta.content is not None:
                yield f"data: {delta.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _security_enabled(request: Request) -> bool:
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "security_enabled", True))


def _validate_chat_input(req: ChatRequest) -> None:
    for message in req.messages:
        if message.role != "user":
            continue
        result = validate_input(message.content)
        if not result.ok:
            # Support Copilot blocks suspicious prompts at the API boundary.
            # Returning a safe fallback would still hide an active attack from clients and garak.
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "security_input_blocked",
                    "message": "Request blocked by input security policy",
                    "rule": result.rule,
                },
            )


def _security_context(request: Request) -> tuple[str, str]:
    canary = getattr(request.app.state, "canary", "CANARY_TEST")
    system_prompt = getattr(request.app.state, "security_system_prompt", build_security_system_prompt(canary))
    return system_prompt, canary


def _with_security_system_prompt(req: ChatRequest, system_prompt: str) -> ChatRequest:
    return req.model_copy(
        update={
            "messages": [
                ChatMessage(role="system", content=system_prompt),
                *req.messages,
            ]
        }
    )
