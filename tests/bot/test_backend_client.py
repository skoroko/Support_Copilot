from __future__ import annotations

import json
from uuid import UUID

import httpx
import pytest

from bot.services.backend_client import BackendClient


INTERNAL_TOKEN = "test-internal-token-0000000000000001"


def build_client(http_client: httpx.AsyncClient) -> BackendClient:
    return BackendClient("http://backend", client=http_client, internal_token=INTERNAL_TOKEN)


def bind_owner(client: BackendClient, chat_id: UUID, owner: str = "telegram-1") -> None:
    client._chat_owners[str(chat_id)] = owner

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
        client = build_client(http_client)

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
        assert request.headers["content-type"].startswith("application/x-www-form-urlencoded")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"type":"token","delta":"Pri"}\n\n'
                b'data: {"type":"token","delta":"vet"}\n\n'
                b'data: {"type":"message","message_id":"00000000-0000-0000-0000-000000000999"}\n\n'
                b'data: {"type":"done"}\n\n'
                b'event: sources\n'
                b'data: {"sources":[{"id":"returns-1","snippet":"14 days","score":0.91}]}\n\n'
            ),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://backend", transport=transport) as http_client:
        client = build_client(http_client)
        bind_owner(client, chat_id)

        chunks = [chunk async for chunk in client.send_message(chat_id, "hello")]

    assert chunks == ["Pri", "vet"]
    assert client.last_response_message_id(chat_id) == UUID("00000000-0000-0000-0000-000000000999")
    assert client.last_response_sources(chat_id) == [
        {"id": "returns-1", "snippet": "14 days", "score": 0.91}
    ]


@pytest.mark.asyncio
async def test_save_feedback_includes_sources_from_last_sse_response() -> None:
    chat_id = UUID("00000000-0000-0000-0000-000000000456")
    message_id = UUID("00000000-0000-0000-0000-000000000999")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith(f"/messages/{message_id}/feedback")
        assert json.loads(request.content) == {
            "value": "up",
            "shown_sources": [{"id": "returns-1", "score": 0.91}],
        }
        return httpx.Response(200, json={"value": "up"})

    async with httpx.AsyncClient(base_url="http://backend", transport=httpx.MockTransport(handler)) as http_client:
        client = build_client(http_client)
        bind_owner(client, chat_id)
        client._last_sources[str(chat_id)] = [{"id": "returns-1", "score": 0.91}]
        response = await client.save_feedback(chat_id, message_id, "up")

    assert response == {"value": "up"}


@pytest.mark.asyncio
async def test_message_sources_are_loaded_from_backend_when_bot_cache_is_empty() -> None:
    chat_id = UUID("00000000-0000-0000-0000-000000000456")
    message_id = UUID("00000000-0000-0000-0000-000000000999")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.method == "GET":
            return httpx.Response(200, json={"sources": [{"id": "returns-1", "score": 0.91}]})
        assert json.loads(request.content) == {
            "value": "down",
            "shown_sources": [{"id": "returns-1", "score": 0.91}],
        }
        return httpx.Response(200, json={"value": "down"})

    async with httpx.AsyncClient(base_url="http://backend", transport=httpx.MockTransport(handler)) as http_client:
        client = build_client(http_client)
        response = await client.save_feedback(
            chat_id,
            message_id,
            "down",
            owner_external_id="telegram-1",
        )

    assert response == {"value": "down"}
    assert calls == [
        f"GET /chats/{chat_id}/messages/{message_id}/sources",
        f"POST /chats/{chat_id}/messages/{message_id}/feedback",
    ]


@pytest.mark.asyncio
async def test_send_message_uses_multipart_for_media() -> None:
    chat_id = UUID("00000000-0000-0000-0000-000000000457")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        body = request.read()
        assert b'name="content"' in body
        assert b"question" in body
        assert b'name="media"; filename="image.png"' in body
        assert b"image-bytes" in body
        return httpx.Response(200, content=b'data: {"type":"done"}\n\n')

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://backend", transport=transport) as http_client:
        client = build_client(http_client)
        bind_owner(client, chat_id)
        chunks = [
            chunk
            async for chunk in client.send_message(
                chat_id, "question", media=b"image-bytes", mime="image/png"
            )
        ]

    assert chunks == []


@pytest.mark.asyncio
async def test_clear_messages_sends_delete_to_chat_messages() -> None:
    chat_id = UUID("00000000-0000-0000-0000-000000000789")
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://backend", transport=transport) as http_client:
        client = build_client(http_client)
        bind_owner(client, chat_id)

        await client.clear_messages(chat_id)

    assert seen == [("DELETE", f"/chats/{chat_id}/messages")]
