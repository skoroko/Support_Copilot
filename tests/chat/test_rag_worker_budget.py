from __future__ import annotations

import asyncio
from threading import Event
from urllib.parse import urlencode

import pytest
from pydantic import SecretStr

from app.chat.repositories.json_repo import JsonChatRepository
from app.core.auth import InMemoryRequestBudget
from app.core.config import get_settings
from app.main import app


class BlockingRAG:
    def __init__(self) -> None:
        self.read_started = Event()
        self.release = Event()
        self.closed = Event()
        self.produced = 0

    def stream_answer(self, question: str, *, history: list[dict]):
        try:
            self.produced += 1
            yield {"type": "delta", "delta": "first"}
            self.read_started.set()
            self.release.wait(timeout=5)
            for _ in range(100):
                self.produced += 1
                yield {"type": "delta", "delta": "tail"}
        finally:
            self.closed.set()


def _scope(path: str, *, method: str = "GET", body: bytes = b"") -> dict:
    headers = [
        (b"x-internal-token", b"test-internal-token-0000000000000001"),
        (b"x-user-id", b"same-principal"),
        (b"x-user-interface", b"cli"),
    ]
    if body:
        headers.extend([
            (b"content-type", b"application/x-www-form-urlencoded"),
            (b"content-length", str(len(body)).encode()),
        ])
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


async def _request(path: str) -> tuple[int, bytes]:
    messages: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    await app(_scope(path), receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return start["status"], body


@pytest.mark.asyncio
async def test_asgi_disconnect_keeps_budget_slot_until_sync_rag_worker_exits(tmp_path) -> None:
    settings = get_settings().model_copy(update={
        "internal_token": SecretStr("test-internal-token-0000000000000001"),
        "api_rate_limit_per_minute": 20,
        "api_max_concurrency_per_principal": 1,
        "chat_repository": "json",
        "chat_storage_dir": tmp_path,
        "moderation_enabled": False,
    })
    repository = JsonChatRepository(tmp_path)
    chat = await repository.create_chat("same-principal", "cli")
    rag = BlockingRAG()
    budget = InMemoryRequestBudget(requests_per_minute=20, max_concurrency=1)
    old_budget = getattr(app.state, "request_budget", None)
    old_rag = getattr(app.state, "rag_service", None)
    old_openai = getattr(app.state, "openai", None)

    app.dependency_overrides[get_settings] = lambda: settings
    app.state.request_budget = budget
    app.state.rag_service = rag
    app.state.openai = object()
    disconnect = asyncio.Event()
    first_body = asyncio.Event()
    request_body = urlencode({"content": "question"}).encode()
    sent_request = False

    async def receive_stream() -> dict:
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": request_body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send_stream(message: dict) -> None:
        if message["type"] == "http.response.body" and b'"delta": "first"' in message.get("body", b""):
            first_body.set()

    task = asyncio.create_task(app(
        _scope(f"/chats/{chat.id}/messages", method="POST", body=request_body),
        receive_stream, send_stream,
    ))
    try:
        try:
            await asyncio.wait_for(first_body.wait(), 2)
            assert await asyncio.to_thread(rag.read_started.wait, 1)
            disconnect.set()
            await asyncio.wait_for(task, 2)
            assert budget._active.get("same-principal") == 1
            status, body = await _request(f"/chats/{chat.id}")
            assert status == 429
            assert b"principal_concurrency_limited" in body
        finally:
            disconnect.set()
            rag.release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        for _ in range(100):
            if rag.closed.is_set() and not budget._active:
                break
            await asyncio.sleep(0.01)
        assert rag.closed.is_set()
        assert rag.produced < 100
        assert not budget._active
        status, _ = await _request(f"/chats/{chat.id}")
        assert status == 200
    finally:
        app.dependency_overrides.clear()
        app.state.request_budget = old_budget
        app.state.rag_service = old_rag
        app.state.openai = old_openai
