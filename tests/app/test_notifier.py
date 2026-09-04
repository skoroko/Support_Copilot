from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from app.services.notifier import notify_user


@pytest.mark.asyncio
async def test_notifier_sends_internal_token_and_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/notify"
        assert request.headers["X-Internal-Token"] == "secret"
        assert request.content == b'{"chat_id":42,"text":"done"}'
        return httpx.Response(200, json={"ok": True})

    settings = SimpleNamespace(bot_url="http://bot:9000", internal_token=SecretStr("secret"))
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await notify_user(42, "done", settings=settings, client=client)
