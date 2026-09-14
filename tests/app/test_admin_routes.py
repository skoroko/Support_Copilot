from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.admin.deps import get_admin_service
from app.admin.models import BroadcastItem, StatsOut, UserOut
from app.core.config import get_settings
from app.main import app


class FakeAdminService:
    def __init__(self) -> None:
        self.broadcasts: list[BroadcastItem] = []

    async def stats(self) -> StatsOut:
        return StatsOut(total_messages=8, active_users=2, avg_latency_ms=125.5, moderation_block_rate=0.25, feedback_up_ratio=0.75)

    async def users(self, *, limit: int, interface: str | None = None) -> list[UserOut]:
        assert limit <= 1000
        return [UserOut(owner_external_id="42", chat_count=2, last_seen_at=datetime.now(timezone.utc))]

    async def enqueue_broadcast(self, message: str, interface: str) -> BroadcastItem:
        item = BroadcastItem(message=message, interface=interface)
        self.broadcasts.append(item)
        return item

    async def pending_broadcasts(self, *, limit: int) -> list[BroadcastItem]:
        return self.broadcasts[:limit]

    async def update_broadcast_status(self, broadcast_id, status: str):
        for item in self.broadcasts:
            if item.id == broadcast_id:
                return item.model_copy(update={"status": status})
        return None


def test_admin_routes_require_token_and_return_data() -> None:
    fake = FakeAdminService()
    settings = get_settings().model_copy(update={"admin_token": SecretStr("admin-secret")})
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_admin_service] = lambda: fake
    try:
        with TestClient(app) as client:
            assert client.get("/chats/admin/stats").status_code == 401
            headers = {"X-Admin-Token": "admin-secret"}
            stats = client.get("/chats/admin/stats", headers=headers)
            users = client.get("/chats/admin/users?limit=10", headers=headers)
            broadcast = client.post(
                "/chats/admin/broadcast",
                headers=headers,
                json={"message": "Maintenance", "interface_filter": "telegram"},
            )
    finally:
        app.dependency_overrides.clear()

    assert stats.status_code == 200
    assert stats.json()["moderation_block_rate"] == 0.25
    assert users.json()[0]["owner_external_id"] == "42"
    assert broadcast.status_code == 200
    assert broadcast.json()["status"] == "pending"
