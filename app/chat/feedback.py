from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


FeedbackValue = Literal["up", "down"]


class FeedbackIn(BaseModel):
    value: FeedbackValue
    shown_sources: list[dict[str, Any]] = Field(default_factory=list, max_length=10)


class MessageFeedback(BaseModel):
    message_id: UUID
    owner_external_id: str
    value: FeedbackValue
    shown_sources: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackTargetNotFoundError(ValueError):
    pass


class FeedbackService:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    async def save(
        self,
        chat_id: UUID,
        message_id: UUID,
        value: FeedbackValue,
        shown_sources: list[dict[str, Any]] | None = None,
    ) -> MessageFeedback:
        chat = await self.repository.get_chat(chat_id)
        if chat is None:
            raise FeedbackTargetNotFoundError(f"Chat {chat_id} not found")
        message = await self.repository.get_message(chat_id, message_id)
        if message is None or message.role != "assistant":
            raise FeedbackTargetNotFoundError(f"Assistant message {message_id} not found")
        return await self.repository.save_feedback(
            MessageFeedback(
                message_id=message_id,
                owner_external_id=chat.owner_external_id,
                value=value,
                shown_sources=list(shown_sources or []),
            )
        )
