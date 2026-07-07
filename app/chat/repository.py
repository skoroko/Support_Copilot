from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.chat.domain import Chat, ChatMessage


class ChatRepository(Protocol):
    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        ...

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        ...

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        ...

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        ...

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        ...

