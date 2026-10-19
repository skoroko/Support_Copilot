from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator
from starlette.datastructures import FormData, UploadFile

from app.admin.auth import require_admin
from app.chat.agent import StaleApprovalError
from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage, telegram_chat_id_from_owner
from app.chat.expansion import ExpansionUnavailableError
from app.chat.feedback import FeedbackIn, FeedbackTargetNotFoundError, MessageFeedback
from app.chat.media import (
    MediaExtractionError,
    MediaProcessingError,
    MediaTooLargeError,
    UnsupportedMediaTypeError,
)
from app.chat.service import (
    ChatNotFoundError,
    ModerationBlockedError,
    StreamReplace,
    media_parts,
)
from app.core.auth import PrincipalDep
from app.core.sse import with_heartbeats
from app.core.upload_boundary import CHAT_MEDIA_MAX_FILES
from app.deps.providers import SettingsDep
from app.observability.logging import get_logger
from app.services.notifier import notify_user
from app.services.turn_budget import AgentTurnTimeout
from app.services.security.input_validator import (
    MAX_MESSAGE_CHARS,
    decoded_fragments,
    validate_input,
)

router = APIRouter(tags=["chats"])
logger = get_logger(__name__)


class _ClosingStreamingResponse(StreamingResponse):
    """Close the SSE generator and release its request slot on disconnect."""

    def __init__(
        self,
        content: AsyncIterator[str],
        *,
        on_close: Callable[[], Awaitable[None]] | None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(content, media_type="text/event-stream", headers=headers)
        self._on_close = on_close

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                close = getattr(self.body_iterator, "aclose", None)
                if callable(close):
                    await close()
            finally:
                if self._on_close is not None:
                    await self._on_close()

RAG_UNAVAILABLE_NOTICE = (
    "База знаний временно недоступна — ответ дан без опоры на документацию. "
    "Проверьте критичные детали или передайте обращение специалисту."
)


class CreateChatOut(BaseModel):
    chat_id: UUID


class SystemMessageIn(BaseModel):
    text: str = Field(min_length=1)
    notify: bool = False


class ApprovalIn(BaseModel):
    approved: bool


class ExpansionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmed: StrictBool

    @field_validator("confirmed")
    @classmethod
    def require_confirmation(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Explicit confirmation is required")
        return value


def _approval_preview(draft: dict | None) -> dict | None:
    if not draft:
        return None
    return {key: draft[key] for key in ("approval_id", "chat_id", "text")}


async def _authenticated_message_form(
    request: Request,
    _principal: PrincipalDep,
) -> AsyncIterator[FormData]:
    # Resolve service auth/principal and budget before parsing multipart.
    async with request.form(max_files=CHAT_MEDIA_MAX_FILES, max_fields=1) as form:
        yield form


@router.post("/chats", response_model=CreateChatOut)
async def create_chat(chat_service: ChatServiceDep, principal: PrincipalDep) -> CreateChatOut:
    chat = await chat_service.create_chat(
        owner_external_id=principal.user_id,
        interface=principal.interface,
    )
    return CreateChatOut(chat_id=chat.id)


@router.get("/chats/{chat_id}", response_model=Chat)
async def get_chat(chat_id: UUID, chat_service: ChatServiceDep, principal: PrincipalDep) -> Chat:
    return await _owned_chat(chat_id, chat_service, principal)


@router.get("/chats/{chat_id}/messages", response_model=list[ChatMessage])
async def list_messages(
    chat_id: UUID,
    chat_service: ChatServiceDep,
    principal: PrincipalDep,
    limit: int = 50,
) -> list[ChatMessage]:
    await _owned_chat(chat_id, chat_service, principal)
    return await chat_service.list_messages(chat_id, limit)


@router.post(
    "/chats/{chat_id}/messages",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "minLength": 1},
                            "media": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "maxItems": CHAT_MEDIA_MAX_FILES,
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
        },
    },
)
async def send_message(
    chat_id: UUID,
    request: Request,
    chat_service: ChatServiceDep,
    principal: PrincipalDep,
    form: FormData = Depends(_authenticated_message_form),
) -> StreamingResponse:
    content = form.get("content")
    if not isinstance(content, str) or not content:
        raise HTTPException(status_code=422, detail={"code": "content_required", "message": "Content is required"})
    media = form.getlist("media")
    if any(not isinstance(item, UploadFile) for item in media):
        raise HTTPException(status_code=422, detail={"code": "invalid_media", "message": "Media must be files"})
    decoded = _validate_user_content(content)
    await _owned_chat(chat_id, chat_service, principal)

    media_refs = None
    if media:
        parts: list[dict] = []
        files_meta: list[dict] = []
        for item in media:
            try:
                part = await chat_service.process_media(item)
            except UnsupportedMediaTypeError as exc:
                raise HTTPException(status_code=415, detail={"code": "unsupported_media_type", "message": str(exc)}) from exc
            except MediaTooLargeError as exc:
                raise HTTPException(status_code=413, detail={"code": "media_too_large", "message": str(exc)}) from exc
            except (MediaExtractionError, MediaProcessingError) as exc:
                raise HTTPException(status_code=422, detail={"code": "media_processing_failed", "message": str(exc)}) from exc
            parts.append(part)
            files_meta.append({"mime": item.content_type, "size": item.size, "filename": item.filename})
        media_refs = {
            "mime": files_meta[0]["mime"],
            "size": files_meta[0]["size"],
            "filename": files_meta[0]["filename"],
            "part": parts[0],
            "parts": parts,
            "files": files_meta,
        }

    media_texts = [
        str(part.get("text", ""))
        for part in media_parts(media_refs)
        if part.get("type") == "text" and part.get("text")
    ]
    moderation_content = "\n".join([content, *decoded, *media_texts]).strip()
    ensure_input_allowed = getattr(chat_service, "ensure_input_allowed", None)
    input_moderated = ensure_input_allowed is not None
    if ensure_input_allowed is not None:
        try:
            await ensure_input_allowed(chat_id, moderation_content)
        except ModerationBlockedError as exc:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "moderation_blocked",
                    "categories": exc.result.categories,
                    "blocked_by": exc.result.blocked_by,
                },
            ) from exc

    async def events():
        try:
            message_kwargs = {}
            if media_refs is not None:
                message_kwargs["media_refs"] = media_refs
            if input_moderated:
                message_kwargs["input_moderated"] = True
            rag_service = getattr(request.app.state, "rag_service", None)
            if rag_service is not None:
                message_kwargs["rag_service"] = rag_service
            elif _rag_enabled(request) and getattr(chat_service, "agent", None) is None:
                notice = json.dumps(
                    {
                        "type": "notice",
                        "code": "rag_unavailable",
                        "message": RAG_UNAVAILABLE_NOTICE,
                    },
                    ensure_ascii=False,
                )
                yield f"data: {notice}\n\n"
            stream = chat_service.send_message(chat_id, content, **message_kwargs)
            async with aclosing(stream):
                async for chunk in stream:
                    if isinstance(chunk, StreamReplace):
                        payload = json.dumps({"type": "replace", "text": chunk.text}, ensure_ascii=False)
                    else:
                        payload = json.dumps({"type": "token", "delta": chunk}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
            latest = await chat_service.list_messages(chat_id, limit=1)
            message_id = str(latest[-1].id) if latest and latest[-1].role == "assistant" else None
            if message_id is not None:
                metadata = json.dumps({"type": "message", "message_id": message_id,
                                       "answer_info": (latest[-1].media_refs or {}).get("answer", {})}, ensure_ascii=False)
                yield f"data: {metadata}\n\n"
            yield 'data: {"type":"done"}\n\n'
            sources = _last_response_sources(chat_service, chat_id)
            sources_payload = json.dumps({"sources": sources}, ensure_ascii=False)
            yield f"event: sources\ndata: {sources_payload}\n\n"
            approval = getattr(chat_service, "last_approval", lambda _id: None)(chat_id)
            if approval:
                preview = json.dumps({"type": "approval", "approval": _approval_preview(approval)}, ensure_ascii=False)
                yield f"data: {preview}\n\n"
        except asyncio.CancelledError:
            raise
        except (AgentTurnTimeout, TimeoutError):
            logger.warning("chat_turn_timeout")
            payload = json.dumps({"type": "error", "code": "chat_turn_timeout",
                "message": "Не удалось завершить ответ за отведённое время. Сервис модели или поиска отвечает слишком долго. Повторите запрос позже или передайте обращение специалисту."},
                ensure_ascii=False)
            yield f"event: error\ndata: {payload}\n\n"
            return
        except Exception as exc:  # noqa: BLE001 - response headers have already been sent
            # Provider exception messages may contain credentials or request
            # bodies. Record only the class so SSE failures are diagnosable.
            logger.warning("chat_stream_failed", error_type=type(exc).__name__)
            payload = json.dumps(
                {
                    "type": "error",
                    "code": "chat_stream_failed",
                    "message": "Не удалось завершить ответ. Повторите запрос или передайте обращение специалисту.",
                },
                ensure_ascii=False,
            )
            yield f"event: error\ndata: {payload}\n\n"
            return

    lease = getattr(request.state, "request_budget_lease", None)
    release_stream = lease.hold() if lease is not None else None
    return _ClosingStreamingResponse(
        with_heartbeats(events()),
        on_close=release_stream,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chats/{chat_id}/approval")
async def pending_approval(chat_id: UUID, chat_service: ChatServiceDep, principal: PrincipalDep) -> dict:
    await _owned_chat(chat_id, chat_service, principal)
    return {"approval": _approval_preview(await chat_service.pending_approval(chat_id))}


@router.get("/chats/{chat_id}/messages/{message_id}/expansion")
async def preview_expansion(chat_id: UUID, message_id: UUID, chat_service: ChatServiceDep,
                            principal: PrincipalDep) -> dict:
    await _owned_chat(chat_id, chat_service, principal)
    try:
        return await chat_service.expansion_availability(chat_id, message_id)
    except ExpansionUnavailableError as exc:
        raise HTTPException(status_code=409, detail={"code": "expansion_unavailable", "message": str(exc)}) from exc


@router.post("/chats/{chat_id}/messages/{message_id}/expand")
async def confirm_expansion(chat_id: UUID, message_id: UUID, payload: ExpansionIn,
                            chat_service: ChatServiceDep, principal: PrincipalDep) -> dict:
    await _owned_chat(chat_id, chat_service, principal)
    try:
        return await chat_service.expand_response(chat_id, message_id)
    except ExpansionUnavailableError as exc:
        raise HTTPException(status_code=409, detail={"code": "expansion_unavailable", "message": str(exc)}) from exc
    except ModerationBlockedError as exc:
        raise HTTPException(status_code=403, detail={"code": "moderation_blocked"}) from exc


@router.post("/chats/{chat_id}/approvals/{approval_id}")
async def resume_approval(chat_id: UUID, approval_id: UUID, payload: ApprovalIn,
                          chat_service: ChatServiceDep, principal: PrincipalDep) -> dict:
    await _owned_chat(chat_id, chat_service, principal)
    try:
        result = await chat_service.approve(chat_id, approval_id.hex, payload.approved)
    except StaleApprovalError as exc:
        raise HTTPException(status_code=409, detail={"code": "stale_approval", "message": str(exc)}) from exc
    except ModerationBlockedError as exc:
        raise HTTPException(status_code=403, detail={"code": "moderation_blocked"}) from exc
    result["approval"] = _approval_preview(result.get("approval"))
    return result


@router.get("/chats/{chat_id}/messages/{message_id}/sources")
async def get_response_sources(
    chat_id: UUID,
    message_id: UUID,
    chat_service: ChatServiceDep,
    principal: PrincipalDep,
) -> dict[str, list[dict]]:
    await _owned_chat(chat_id, chat_service, principal)
    try:
        return {"sources": await chat_service.response_sources(chat_id, message_id)}
    except FeedbackTargetNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "feedback_target_not_found", "message": str(exc)},
        ) from exc


@router.post(
    "/chats/{chat_id}/messages/{message_id}/feedback",
    response_model=MessageFeedback,
)
async def save_feedback(
    chat_id: UUID,
    message_id: UUID,
    payload: FeedbackIn,
    chat_service: ChatServiceDep,
    principal: PrincipalDep,
) -> MessageFeedback:
    await _owned_chat(chat_id, chat_service, principal)
    try:
        if payload.shown_sources:
            return await chat_service.submit_feedback(
                chat_id,
                message_id,
                payload.value,
                payload.shown_sources,
            )
        return await chat_service.submit_feedback(chat_id, message_id, payload.value)
    except FeedbackTargetNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "feedback_target_not_found", "message": str(exc)},
        ) from exc


@router.post("/chats/admin/{chat_id}/notification")
async def add_system_message(
    chat_id: UUID,
    payload: SystemMessageIn,
    chat_service: ChatServiceDep,
    settings: SettingsDep,
    _admin: None = Depends(require_admin),
) -> dict[str, str]:
    try:
        chat = await chat_service.append_system_message(chat_id, payload.text)
    except ChatNotFoundError as exc:
        raise _chat_not_found(chat_id) from exc

    if payload.notify:
        try:
            telegram_chat_id = telegram_chat_id_from_owner(chat.owner_external_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_external_id", "message": "Chat owner_external_id is not a Telegram chat id."},
            ) from exc
        await notify_user(telegram_chat_id, payload.text, settings=settings)
    return {"status": "ok"}


@router.delete("/chats/{chat_id}/messages")
async def clear_messages(
    chat_id: UUID,
    chat_service: ChatServiceDep,
    principal: PrincipalDep,
) -> dict[str, str]:
    await _owned_chat(chat_id, chat_service, principal)
    await chat_service.clear_history(chat_id)
    return {"status": "ok"}


def _rag_enabled(request: Request) -> bool:
    """Whether the deployment expects a working RAG service.

    ``rag_enabled=False`` is a deliberate deployment choice, not an outage, so
    only the enabled-but-missing state earns the user-facing notice.
    """

    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "rag_enabled", False))


def _chat_not_found(chat_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"code": "chat_not_found", "message": f"Chat {chat_id} not found"},
    )


async def _owned_chat(chat_id: UUID, chat_service: object, principal: object) -> Chat:
    chat = await chat_service.get_chat(chat_id)  # type: ignore[attr-defined]
    if (
        chat is None
        or chat.owner_external_id != principal.user_id  # type: ignore[attr-defined]
        or chat.interface != principal.interface  # type: ignore[attr-defined]
    ):
        raise _chat_not_found(chat_id)
    return chat


def _validate_user_content(content: str) -> list[str]:
    result = validate_input(content, max_chars=MAX_MESSAGE_CHARS)
    if result.ok:
        return decoded_fragments(content)
    detail: dict[str, object] = {
        "code": "security_input_blocked",
        "message": result.reason or "input blocked by security policy",
        "rule": result.rule,
    }
    if result.rule == "length":
        detail["limit"] = MAX_MESSAGE_CHARS
    raise HTTPException(status_code=400, detail=detail)


def _last_response_sources(chat_service: object, chat_id: UUID) -> list[dict]:
    get_sources = getattr(chat_service, "last_response_sources", None)
    if not callable(get_sources):
        return []
    sources = get_sources(chat_id)
    if not isinstance(sources, list):
        return []
    return [dict(source) for source in sources if isinstance(source, dict)]
