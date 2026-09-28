from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.auth import PrincipalDep
from app.core.exceptions import LLMError
from app.deps.providers import LLMServiceDep
from app.observability.logging import get_logger
from app.schemas.chat import ChatRequest, ChatResponse
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
logger = get_logger("llm-service")


def _sse_data(payload: str) -> str:
    """Encode text as an SSE data event, including embedded newlines."""

    normalized = payload.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(f"data: {line}\n" for line in normalized.split("\n")) + "\n"


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Complete one chat request",
    responses=RESPONSES,
)
async def complete_chat(
    req: ChatRequest,
    service: LLMServiceDep,
    request: Request,
    _principal: PrincipalDep,
) -> ChatResponse:
    _validate_model(req, request)
    if not _security_enabled(request):
        return await service.complete(req, security_enabled=False)

    _validate_chat_input(req)
    system_prompt, canary = _security_context(request)
    return await service.complete(req, system_prompt=system_prompt, canary=canary)


@router.post(
    "/chat/stream",
    summary="Stream chat completion as Server-Sent Events",
    responses=RESPONSES,
)
async def stream_chat(
    req: ChatRequest,
    service: LLMServiceDep,
    request: Request,
    _principal: PrincipalDep,
) -> StreamingResponse:
    _validate_model(req, request)
    if not _security_enabled(request):
        async def insecure_events() -> AsyncIterator[str]:
            try:
                async for delta in service.stream(req.model_copy(update={"stream": True}), security_enabled=False):
                    if delta.usage is not None:
                        yield _sse_data(json.dumps({'usage': delta.usage.model_dump()}, ensure_ascii=False))
                    elif delta.content is not None:
                        yield _sse_data(delta.content)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - terminal SSE boundary
                logger.warning("chat_stream_failed", error_type=type(exc).__name__)
                yield _sse_error(exc)
                return
            yield "data: [DONE]\n\n"

        return StreamingResponse(insecure_events(), media_type="text/event-stream")

    _validate_chat_input(req)
    system_prompt, canary = _security_context(request)

    async def events() -> AsyncIterator[str]:
        try:
            async for delta in service.stream(
                req.model_copy(update={"stream": True}),
                system_prompt=system_prompt,
                canary=canary,
            ):
                if delta.usage is not None:
                    yield _sse_data(json.dumps({'usage': delta.usage.model_dump()}, ensure_ascii=False))
                elif delta.content is not None:
                    yield _sse_data(delta.content)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - terminal SSE boundary
            logger.warning("chat_stream_failed", error_type=type(exc).__name__)
            yield _sse_error(exc)
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _security_enabled(request: Request) -> bool:
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "security_enabled", True))


def _validate_chat_input(req: ChatRequest) -> None:
    for message in req.messages:
        result = validate_input(message.content)
        if not result.ok:
            # Support Copilot blocks suspicious prompts at the API boundary.
            # Returning a safe fallback would still hide an active attack from clients and garak.
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "security_input_blocked",
                    "message": result.reason or "Request blocked by input security policy",
                    "rule": result.rule,
                },
            )


def _security_context(request: Request) -> tuple[str, str]:
    canary = getattr(request.app.state, "canary", "CANARY_TEST")
    system_prompt = getattr(request.app.state, "security_system_prompt", build_security_system_prompt(canary))
    return system_prompt, canary


def _validate_model(req: ChatRequest, request: Request) -> None:
    settings = request.app.state.settings
    model = req.model or settings.default_model
    if model not in settings.effective_allowed_models:
        raise HTTPException(
            status_code=422,
            detail={"code": "model_not_allowed", "message": "Requested model is not allowed."},
        )


def _sse_error(exc: Exception) -> str:
    code = exc.code if isinstance(exc, LLMError) else "chat_stream_failed"
    payload = {
        "type": "error",
        "code": code,
        "message": "Не удалось завершить ответ. Повторите запрос или передайте обращение специалисту.",
    }
    return f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
