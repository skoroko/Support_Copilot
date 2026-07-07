from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage
from app.chat.service import ChatNotFoundError
from app.services.security.input_validator import validate_input


router = APIRouter(tags=["chats"])


class CreateChatIn(BaseModel):
    owner_external_id: str = Field(min_length=1)
    interface: str = Field(min_length=1)
    system_prompt: str | None = None


class CreateChatOut(BaseModel):
    chat_id: UUID


class MessageIn(BaseModel):
    content: str = Field(min_length=1)


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
async def send_message(chat_id: UUID, payload: MessageIn, chat_service: ChatServiceDep) -> StreamingResponse:
    _validate_user_content(payload.content)
    if await chat_service.get_chat(chat_id) is None:
        raise _chat_not_found(chat_id)

    async def events():
        try:
            async for chunk in chat_service.send_message(chat_id, payload.content):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except ChatNotFoundError as exc:
            raise _chat_not_found(chat_id) from exc

    return StreamingResponse(events(), media_type="text/event-stream")


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
