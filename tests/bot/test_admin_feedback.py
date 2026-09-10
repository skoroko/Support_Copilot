from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest

from bot.handlers._common import backend_error_message
from bot.handlers.admin import IsAdmin
from bot.keyboards.inline import answer_kb, decode_message_context, feedback_kb
from bot.services.backend_client import BackendClient


@pytest.mark.asyncio
async def test_is_admin_uses_router_level_id_set() -> None:
    admin = SimpleNamespace(from_user=SimpleNamespace(id=42))
    stranger = SimpleNamespace(from_user=SimpleNamespace(id=7))
    check = IsAdmin([42])

    assert await check(admin) is True
    assert await check(stranger) is False


def test_feedback_keyboard_uses_required_callback_contract() -> None:
    message_id = UUID("00000000-0000-0000-0000-000000000123")
    markup = feedback_kb(message_id)
    callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
    assert callbacks == [f"fb:up:{message_id}", f"fb:down:{message_id}"]


def test_answer_keyboard_compacts_chat_and_message_ids_below_telegram_limit() -> None:
    chat_id = UUID("00000000-0000-0000-0000-000000000111")
    message_id = UUID("00000000-0000-0000-0000-000000000123")
    markup = answer_kb(chat_id, message_id, source_count=10)
    callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
    texts = [button.text for row in markup.inline_keyboard for button in row]

    assert len(callbacks) == 3
    assert texts[0] == "📚 Документы (10)"
    assert all(callback is not None and len(callback.encode("utf-8")) <= 64 for callback in callbacks)
    assert callbacks[0].startswith("src:")
    assert decode_message_context(callbacks[0].removeprefix("src:")) == (chat_id, message_id)
    assert callbacks[1].startswith("fb:u:")
    assert callbacks[2].startswith("fb:d:")


def test_moderation_403_has_user_friendly_message() -> None:
    request = httpx.Request("POST", "http://backend/chats/x/messages")
    response = httpx.Response(
        403,
        request=request,
        json={"error": {"code": "moderation_blocked", "categories": ["violence"]}},
    )
    message = backend_error_message(httpx.HTTPStatusError("blocked", request=request, response=response))
    assert "правил" in message


@pytest.mark.asyncio
async def test_admin_client_sends_admin_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Admin-Token"] == "secret"
        return httpx.Response(200, json={"total_messages": 0})

    async with httpx.AsyncClient(base_url="http://backend", transport=httpx.MockTransport(handler)) as http:
        client = BackendClient("http://backend", client=http, admin_token="secret")
        assert await client.get_admin_stats() == {"total_messages": 0}
