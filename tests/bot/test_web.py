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
    assert accepted.json() == {"ok": True}
    bot.send_message.assert_awaited_once_with(chat_id=42, text="Готово")
