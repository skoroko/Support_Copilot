from __future__ import annotations

from uuid import UUID

import httpx
import pytest

from bot.services.backend_client import BackendClient


@pytest.mark.asyncio
async def test_get_or_create_chat_returns_uuid_and_is_idempotent() -> None:
    chat_id = UUID("00000000-0000-0000-0000-000000000123")
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "POST"
        assert request.url.path == "/chats"
        return httpx.Response(200, json={"chat_id": str(chat_id)})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://backend", transport=transport) as http_client:
        client = BackendClient("http://backend", client=http_client)

        first = await client.get_or_create_chat("telegram-1", "telegram")
        second = await client.get_or_create_chat("telegram-1", "telegram")

    assert first == chat_id
    assert second == chat_id
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_send_message_parses_sse_frames() -> None:
    chat_id = UUID("00000000-0000-0000-0000-000000000456")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/chats/{chat_id}/messages"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: Pri\n\ndata: vet\n\ndata: [DONE]\n\n",
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://backend", transport=transport) as http_client:
        client = BackendClient("http://backend", client=http_client)

        chunks = [chunk async for chunk in client.send_message(chat_id, "hello")]

    assert chunks == ["Pri", "vet"]


@pytest.mark.asyncio
async def test_clear_messages_sends_delete_to_chat_messages() -> None:
    chat_id = UUID("00000000-0000-0000-0000-000000000789")
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://backend", transport=transport) as http_client:
        client = BackendClient("http://backend", client=http_client)

        await client.clear_messages(chat_id)

    assert seen == [("DELETE", f"/chats/{chat_id}/messages")]
