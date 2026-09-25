from __future__ import annotations

import asyncio
import json
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.admin.auth import require_admin
from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage, telegram_chat_id_from_owner
from app.chat.feedback import FeedbackIn, FeedbackTargetNotFoundError, MessageFeedback
from app.chat.media import MediaExtractionError, MediaProcessingError, MediaTooLargeError, UnsupportedMediaTypeError
from app.chat.service import ChatNotFoundError, ModerationBlockedError, StreamReplace, media_parts
from app.core.auth import PrincipalDep
from app.deps.providers import SettingsDep
from app.services.notifier import notify_user
from app.services.security.input_validator import MAX_MESSAGE_CHARS, decoded_fragments, validate_input


router = APIRouter(tags=["chats"])

RAG_UNAVAILABLE_NOTICE = (
    "База знаний временно недоступна — ответ дан без опоры на документацию. "
    "Проверьте критичные детали или передайте обращение специалисту."
)


class CreateChatOut(BaseModel):
    chat_id: UUID


class SystemMessageIn(BaseModel):
    text: str = Field(min_length=1)
    notify: bool = False


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


@router.post("/chats/{chat_id}/messages")
async def send_message(
    chat_id: UUID,
    request: Request,
    chat_service: ChatServiceDep,
    principal: PrincipalDep,
    content: str = Form(..., min_length=1),
    media: list[UploadFile] | None = File(None),
) -> StreamingResponse:
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
            elif _rag_enabled(request):
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
            async for chunk in stream:
                if isinstance(chunk, StreamReplace):
                    payload = json.dumps({"type": "replace", "text": chunk.text}, ensure_ascii=False)
                else:
                    payload = json.dumps({"type": "token", "delta": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            latest = await chat_service.list_messages(chat_id, limit=1)
            message_id = str(latest[-1].id) if latest and latest[-1].role == "assistant" else None
            if message_id is not None:
                metadata = json.dumps({"type": "message", "message_id": message_id}, ensure_ascii=False)
                yield f"data: {metadata}\n\n"
            yield 'data: {"type":"done"}\n\n'
            sources = _last_response_sources(chat_service, chat_id)
            sources_payload = json.dumps({"sources": sources}, ensure_ascii=False)
            yield f"event: sources\ndata: {sources_payload}\n\n"
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - response headers have already been sent
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

    return StreamingResponse(events(), media_type="text/event-stream")


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
