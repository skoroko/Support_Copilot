from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    chat_id: UUID
    role: Literal["user", "assistant", "system"]
    content: str
    media_refs: dict[str, Any] | None = None
    rag_sources: list[dict[str, Any]] = Field(default_factory=list)
    tokens: int | None = None
    latency_ms: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Chat(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    owner_external_id: str
    interface: str
    system_prompt: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def telegram_chat_id_from_owner(owner_external_id: str) -> int:
    """Telegram delivery chat encoded in a bot ``owner_external_id`` key.

    Bot owner keys are ``"<chat_id>"`` for private dialogs and
    ``"<chat_id>:<sender_id>"`` for group members, so the leading segment is
    always the Telegram chat id the answer should be delivered to.
    """

    return int(owner_external_id.split(":", 1)[0])
