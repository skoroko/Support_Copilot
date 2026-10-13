from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.admin.deps import get_admin_service
from app.admin.models import BroadcastItem, StatsOut, UserOut
from app.admin.service import AdminService
from app.chat.repositories.json_repo import JsonChatRepository
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


@pytest.mark.asyncio
async def test_admin_broadcast_routes_guard_recipient_progress(tmp_path) -> None:
    repository = JsonChatRepository(tmp_path)
    await repository.create_chat("42", "telegram")
    settings = get_settings().model_copy(update={"admin_token": SecretStr("admin-secret")})
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_admin_service] = lambda: AdminService(repository)
    headers = {"X-Admin-Token": "admin-secret"}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            created = await client.post(
                "/chats/admin/broadcast", headers=headers,
                json={"message": "Maintenance", "interface_filter": "telegram"},
            )
            assert created.status_code == 200
            broadcast_id = created.json()["id"]
            claimed = await client.post(
                "/chats/admin/broadcasts/claim", headers=headers,
                json={"limit": 1, "lease_seconds": 300},
            )
            assert claimed.status_code == 200
            lease_id = claimed.json()[0]["lease_id"]
            assert claimed.json()[0]["recipients"][0]["chat_id"] == 42
            assert (await client.post("/chats/admin/broadcasts/claim", headers=headers, json={})).json() == []
            old_status = await client.post(
                f"/chats/admin/broadcasts/{broadcast_id}/status", headers=headers,
                json={"status": "sent"},
            )
            assert old_status.status_code == 404
            attempt = str(uuid4())
            state_url = f"/chats/admin/broadcasts/{broadcast_id}/recipients/42/state"
            stale = await client.post(
                state_url, headers=headers,
                json={"lease_id": str(uuid4()), "attempt_id": attempt, "status": "in_flight"},
            )
            assert stale.status_code == 409
            started = await client.post(
                state_url, headers=headers,
                json={"lease_id": lease_id, "attempt_id": attempt, "status": "in_flight"},
            )
            assert started.status_code == 200
            sent = await client.post(
                state_url, headers=headers,
                json={"lease_id": lease_id, "attempt_id": attempt, "status": "sent"},
            )
            assert sent.status_code == 200
            finished = await client.post(
                f"/chats/admin/broadcasts/{broadcast_id}/finish", headers=headers,
                json={"lease_id": lease_id},
            )
            assert finished.status_code == 200 and finished.json()["status"] == "sent"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_admin_can_inspect_review_broadcasts_without_delivery_side_effects(tmp_path) -> None:
    repository = JsonChatRepository(tmp_path)
    await repository.create_chat("invalid-owner", "telegram")
    item = await repository.enqueue_broadcast("Maintenance", "telegram")
    claim = (await repository.claim_broadcasts())[0]
    assert claim.lease_id is not None
    await repository.finish_broadcast(item.id, claim.lease_id)
    await repository.create_chat("42", "telegram")
    uncertain = await repository.enqueue_broadcast("Second", "telegram")
    uncertain_claim = (await repository.claim_broadcasts())[0]
    assert uncertain_claim.lease_id is not None
    attempt = uuid4()
    await repository.update_broadcast_recipient(
        uncertain.id, uncertain_claim.lease_id, 42, attempt, "in_flight",
    )
    path = repository._broadcast_file()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["lease_expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    await repository._write_jsonl(path, rows)
    recovered = (await repository.claim_broadcasts())[0]
    assert recovered.recipients[0].status == "unknown"
    assert recovered.status == "processing" and recovered.review_required
    settings = get_settings().model_copy(update={"admin_token": SecretStr("admin-secret")})
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_admin_service] = lambda: AdminService(repository)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/chats/admin/broadcasts/review")).status_code == 401
            headers = {"X-Admin-Token": "admin-secret"}
            listed = await client.get("/chats/admin/broadcasts/review?limit=1&offset=0", headers=headers)
            assert listed.status_code == 200
            assert [entry["id"] for entry in listed.json()] == [str(item.id)]
            assert listed.json()[0]["review_required"] is True
            second_page = await client.get("/chats/admin/broadcasts/review?limit=1&offset=1", headers=headers)
            assert [entry["id"] for entry in second_page.json()] == [str(uncertain.id)]
            assert second_page.json()[0]["recipients"][0]["status"] == "unknown"
            assert (await client.get("/chats/admin/broadcasts/review?limit=101", headers=headers)).status_code == 422
            detail = await client.get(f"/chats/admin/broadcasts/{item.id}", headers=headers)
            assert detail.status_code == 200
            assert detail.json()["invalid_recipient_count"] == 1
            assert (await client.get(f"/chats/admin/broadcasts/{uuid4()}", headers=headers)).status_code == 404
    finally:
        app.dependency_overrides.clear()
