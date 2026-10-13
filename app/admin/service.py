from __future__ import annotations

from typing import Any
from uuid import UUID

from app.admin.models import BroadcastItem, StatsOut, UserOut


class AdminService:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    async def stats(self) -> StatsOut:
        return await self.repository.get_admin_stats()

    async def users(self, *, limit: int, interface: str | None = None) -> list[UserOut]:
        return await self.repository.list_admin_users(limit=limit, interface=interface)

    async def enqueue_broadcast(self, message: str, interface: str) -> BroadcastItem:
        return await self.repository.enqueue_broadcast(message, interface)

    async def pending_broadcasts(self, *, limit: int) -> list[BroadcastItem]:
        return await self.repository.list_pending_broadcasts(limit=limit)

    async def review_broadcasts(self, *, limit: int, offset: int) -> list[BroadcastItem]:
        return await self.repository.list_review_broadcasts(limit=limit, offset=offset)

    async def get_broadcast(self, broadcast_id: UUID) -> BroadcastItem | None:
        return await self.repository.get_broadcast(broadcast_id)

    async def claim_broadcasts(self, *, limit: int, lease_seconds: int) -> list[BroadcastItem]:
        return await self.repository.claim_broadcasts(limit=limit, lease_seconds=lease_seconds)

    async def update_broadcast_recipient(
        self, broadcast_id: UUID, lease_id: UUID, chat_id: int, attempt_id: UUID, status: str,
    ) -> BroadcastItem | None:
        return await self.repository.update_broadcast_recipient(
            broadcast_id, lease_id, chat_id, attempt_id, status,
        )

    async def finish_broadcast(self, broadcast_id: UUID, lease_id: UUID) -> BroadcastItem | None:
        return await self.repository.finish_broadcast(broadcast_id, lease_id)
