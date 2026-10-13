from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from app.admin.broadcast_state import snapshot_recipients
from app.chat.repositories.json_repo import JsonChatRepository
from bot.background import broadcast_chat_ids, deliver_broadcast


def test_broadcast_chat_ids_dedupes_group_members() -> None:
    users = [
        {"owner_external_id": "42"},
        {"owner_external_id": "-100:7"},
        {"owner_external_id": "-100:8"},
        {"owner_external_id": "-200:9"},
    ]

    targets, skipped = broadcast_chat_ids(users)

    assert targets == [42, -100, -200]
    assert skipped == 0


def test_broadcast_chat_ids_skips_unparseable_owners() -> None:
    users = [{"owner_external_id": "not-a-number"}, {}, {"owner_external_id": None}]

    targets, skipped = broadcast_chat_ids(users)

    assert targets == []
    assert skipped == 3


def test_broadcast_snapshot_has_no_admin_user_page_limit() -> None:
    recipients, invalid = snapshot_recipients(str(index) for index in range(1, 1003))
    assert len(recipients) == 1002
    assert invalid == 0


class RepositoryBackend:
    def __init__(self, repository: JsonChatRepository) -> None:
        self.repository = repository
        self.fail_sent_once = False

    async def claim_broadcasts(self, *, limit: int = 10, lease_seconds: int = 300) -> list[dict]:
        items = await self.repository.claim_broadcasts(limit, lease_seconds)
        return [item.model_dump(mode="json") for item in items]

    async def update_broadcast_recipient(
        self, broadcast_id: str, lease_id: str, chat_id: int, attempt_id: str, status: str,
    ) -> dict:
        if status == "sent" and self.fail_sent_once:
            self.fail_sent_once = False
            raise httpx.ConnectError("backend unavailable")
        item = await self.repository.update_broadcast_recipient(
            UUID(broadcast_id), UUID(lease_id), chat_id, UUID(attempt_id), status,
        )
        assert item is not None
        return item.model_dump(mode="json")

    async def finish_broadcast(self, broadcast_id: str, lease_id: str) -> dict:
        item = await self.repository.finish_broadcast(UUID(broadcast_id), UUID(lease_id))
        assert item is not None
        return item.model_dump(mode="json")


class RecordingBot:
    def __init__(self) -> None:
        self.sent: list[int] = []

    async def send_message(self, *, chat_id: int, text: str) -> None:
        self.sent.append(chat_id)


@pytest.mark.asyncio
async def test_broadcast_reclaim_does_not_resend_ambiguous_delivery(tmp_path: Path) -> None:
    repository = JsonChatRepository(tmp_path)
    for owner in ("41", "42"):
        await repository.create_chat(owner, "telegram")
    await repository.enqueue_broadcast("Maintenance", "telegram")
    backend = RepositoryBackend(repository)
    bot = RecordingBot()
    first = (await backend.claim_broadcasts())[0]
    backend.fail_sent_once = True

    with pytest.raises(httpx.ConnectError):
        await deliver_broadcast(bot, backend, first)
    assert bot.sent == [41]

    path = repository._broadcast_file()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["lease_expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    await repository._write_jsonl(path, rows)
    reclaimed = (await backend.claim_broadcasts())[0]
    assert [recipient["status"] for recipient in reclaimed["recipients"]] == ["unknown", "pending"]

    completed = await deliver_broadcast(bot, backend, reclaimed)
    assert bot.sent == [41, 42]
    assert completed["status"] == "failed"
    assert completed["review_required"] is True
