from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage
from app.chat.media import MediaExtractionError, MediaProcessingError, UnsupportedMediaTypeError
from app.chat.service import ChatNotFoundError
from app.deps.providers import SettingsDep
from app.services.notifier import notify_user
from app.services.security.input_validator import validate_input


router = APIRouter(tags=["chats"])


class CreateChatIn(BaseModel):
    owner_external_id: str = Field(min_length=1)
    interface: str = Field(min_length=1)
    system_prompt: str | None = None


class CreateChatOut(BaseModel):
    chat_id: UUID


class SystemMessageIn(BaseModel):
    text: str = Field(min_length=1)
    notify: bool = False


@router.post("/chats", response_model=CreateChatOut)
async def create_chat(payload: CreateChatIn, chat_service: ChatServiceDep) -> CreateChatOut:
    chat = await chat_service.create_chat(
        owner_external_id=payload.owner_external_id,
        interface=payload.interface,
        system_prompt=payload.system_prompt,
    )
    return CreateChatOut(chat_id=chat.id)


@router.get("/chats/{chat_id}", response_model=Chat)
async def get_chat(chat_id: UUID, chat_service: ChatServiceDep) -> Chat:
    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise _chat_not_found(chat_id)
    return chat


@router.get("/chats/{chat_id}/messages", response_model=list[ChatMessage])
async def list_messages(chat_id: UUID, chat_service: ChatServiceDep, limit: int = 50) -> list[ChatMessage]:
    if await chat_service.get_chat(chat_id) is None:
        raise _chat_not_found(chat_id)
    return await chat_service.list_messages(chat_id, limit)


@router.post("/chats/{chat_id}/messages")
async def send_message(
    chat_id: UUID,
    chat_service: ChatServiceDep,
    content: str = Form(..., min_length=1),
    media: UploadFile | None = File(None),
) -> StreamingResponse:
    _validate_user_content(content)
    if await chat_service.get_chat(chat_id) is None:
        raise _chat_not_found(chat_id)

    media_refs = None
    if media is not None:
        try:
            part = await chat_service.process_media(media)
        except UnsupportedMediaTypeError as exc:
            raise HTTPException(status_code=415, detail={"code": "unsupported_media_type", "message": str(exc)}) from exc
        except (MediaExtractionError, MediaProcessingError) as exc:
            raise HTTPException(status_code=422, detail={"code": "media_processing_failed", "message": str(exc)}) from exc
        if part.get("type") == "text":
            _validate_user_content(str(part.get("text", "")))
        media_refs = {
            "mime": media.content_type,
            "size": media.size,
            "filename": media.filename,
            "part": part,
        }

    async def events():
        try:
            stream = (
                chat_service.send_message(chat_id, content)
                if media_refs is None
                else chat_service.send_message(chat_id, content, media_refs=media_refs)
            )
            async for chunk in stream:
                payload = json.dumps({"type": "token", "delta": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            yield 'data: {"type":"done"}\n\n'
        except ChatNotFoundError as exc:
            raise _chat_not_found(chat_id) from exc

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/chats/{chat_id}/system-message")
async def add_system_message(
    chat_id: UUID,
    payload: SystemMessageIn,
    chat_service: ChatServiceDep,
    settings: SettingsDep,
) -> dict[str, str]:
    try:
        chat = await chat_service.append_system_message(chat_id, payload.text)
    except ChatNotFoundError as exc:
        raise _chat_not_found(chat_id) from exc

    if payload.notify:
        try:
            telegram_chat_id = int(chat.owner_external_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_external_id", "message": "Chat owner_external_id is not a Telegram chat id."},
            ) from exc
        await notify_user(telegram_chat_id, payload.text, settings=settings)
    return {"status": "ok"}


@router.delete("/chats/{chat_id}/messages")
async def clear_messages(chat_id: UUID, chat_service: ChatServiceDep) -> dict[str, str]:
    if await chat_service.get_chat(chat_id) is None:
        raise _chat_not_found(chat_id)
    await chat_service.clear_history(chat_id)
    return {"status": "ok"}


def _chat_not_found(chat_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"code": "chat_not_found", "message": f"Chat {chat_id} not found"},
    )


def _validate_user_content(content: str) -> None:
    result = validate_input(content)
    if result.ok:
        return
    raise HTTPException(
        status_code=400,
        detail={
            "code": "security_input_blocked",
            "message": result.reason or "input blocked by security policy",
            "rule": result.rule,
        },
    )
