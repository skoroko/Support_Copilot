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

    async def update_broadcast_status(self, broadcast_id: UUID, status: str) -> BroadcastItem | None:
        return await self.repository.update_broadcast_status(broadcast_id, status)
