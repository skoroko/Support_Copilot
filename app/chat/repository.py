from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.admin.models import BroadcastItem, StatsOut, UserOut
from app.chat.domain import Chat, ChatMessage
from app.chat.feedback import MessageFeedback
from app.moderation.models import ModerationResult


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

    async def append_turn(
        self,
        chat_id: UUID,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
    ) -> tuple[ChatMessage, ChatMessage]:
        ...

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        ...

    async def get_message(self, chat_id: UUID, message_id: UUID) -> ChatMessage | None:
        ...

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        ...

    async def save_feedback(self, feedback: MessageFeedback) -> MessageFeedback:
        ...

    async def record_moderation(
        self,
        chat_id: UUID,
        direction: str,
        result: ModerationResult,
    ) -> None:
        ...

    async def get_admin_stats(self) -> StatsOut:
        ...

    async def list_admin_users(self, limit: int = 50, interface: str | None = None) -> list[UserOut]:
        ...

    async def enqueue_broadcast(self, message: str, interface: str) -> BroadcastItem:
        ...

    async def list_pending_broadcasts(self, limit: int = 10) -> list[BroadcastItem]:
        ...

    async def list_review_broadcasts(self, limit: int = 10, offset: int = 0) -> list[BroadcastItem]:
        ...

    async def get_broadcast(self, broadcast_id: UUID) -> BroadcastItem | None:
        ...

    async def claim_broadcasts(self, limit: int = 10, lease_seconds: int = 300) -> list[BroadcastItem]:
        ...

    async def update_broadcast_recipient(
        self, broadcast_id: UUID, lease_id: UUID, chat_id: int, attempt_id: UUID, status: str,
    ) -> BroadcastItem | None:
        ...

    async def finish_broadcast(self, broadcast_id: UUID, lease_id: UUID) -> BroadcastItem | None:
        ...
