from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from bot.web import build_api


def test_notify_requires_internal_token_and_sends_message() -> None:
    bot = SimpleNamespace(send_message=AsyncMock())
    client = TestClient(build_api(bot, "secret"))
    unauthorized = client.post(
        "/notify", headers={"X-Internal-Token": "wrong"}, json={"chat_id": 42, "text": "Готово"}
    )
    accepted = client.post(
        "/notify", headers={"X-Internal-Token": "secret"}, json={"chat_id": 42, "text": "Готово"}
    )
    assert unauthorized.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json() == {"ok": True, "deduplicated": False, "delivery_unknown": False}
    bot.send_message.assert_awaited_once_with(chat_id=42, text="Готово")


def test_notify_deduplicates_claimed_delivery_key() -> None:
    bot = SimpleNamespace(send_message=AsyncMock())
    store = SimpleNamespace(
        set=AsyncMock(side_effect=[True, True, False]),
        get=AsyncMock(return_value="sent"),
    )
    client = TestClient(build_api(bot, "secret", idempotency_store=store))
    payload = {"chat_id": 42, "text": "Готово", "idempotency_key": "stable-key-12345"}

    first = client.post("/notify", headers={"X-Internal-Token": "secret"}, json=payload)
    duplicate = client.post("/notify", headers={"X-Internal-Token": "secret"}, json=payload)

    assert first.json() == {"ok": True, "deduplicated": False, "delivery_unknown": False}
    assert duplicate.json() == {"ok": True, "deduplicated": True, "delivery_unknown": False}
    bot.send_message.assert_awaited_once_with(chat_id=42, text="Готово")
    assert store.set.await_count == 3


def test_notify_marks_ambiguous_failure_unknown_and_blocks_retry() -> None:
    bot = SimpleNamespace(send_message=AsyncMock(side_effect=RuntimeError("telegram unavailable")))
    store = SimpleNamespace(
        set=AsyncMock(side_effect=[True, True, False]),
        get=AsyncMock(return_value="unknown"),
    )
    client = TestClient(build_api(bot, "secret", idempotency_store=store), raise_server_exceptions=False)

    first = client.post(
        "/notify",
        headers={"X-Internal-Token": "secret"},
        json={"chat_id": 42, "text": "Готово", "idempotency_key": "stable-key-12345"},
    )
    retry = client.post(
        "/notify",
        headers={"X-Internal-Token": "secret"},
        json={"chat_id": 42, "text": "Готово", "idempotency_key": "stable-key-12345"},
    )

    assert first.status_code == 502
    assert first.json()["detail"]["code"] == "delivery_unknown"
    assert retry.status_code == 409
    assert retry.json()["detail"]["code"] == "delivery_unknown"
    assert bot.send_message.await_count == 1
