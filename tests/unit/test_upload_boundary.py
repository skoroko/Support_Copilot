from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.datastructures import UploadFile

from app.core.upload_boundary import UploadBoundaryMiddleware
from app.main import app


def test_unauthenticated_document_body_is_not_parsed() -> None:
    with patch.object(UploadFile, "write", wraps=UploadFile.write) as write:
        response = TestClient(app).post(
            "/documents/upload",
            files={"file": ("large.pdf", b"x" * (2 * 1024 * 1024), "application/pdf")},
        )
    assert response.status_code == 401
    write.assert_not_called()


def test_unauthenticated_chat_media_is_not_parsed() -> None:
    with patch.object(UploadFile, "write", wraps=UploadFile.write) as write:
        response = TestClient(app).post(
            "/chats/00000000-0000-0000-0000-000000000000/messages",
            data={"content": "hello"},
            files={"media": ("large.pdf", b"x" * (2 * 1024 * 1024), "application/pdf")},
        )
    assert response.status_code == 401
    write.assert_not_called()


def test_chunked_body_is_bounded_without_content_length(monkeypatch) -> None:
    monkeypatch.setattr("app.core.upload_boundary.DOCUMENT_BODY_LIMIT", 32)
    consumed = 0

    async def downstream(_scope, receive, send):
        nonlocal consumed
        while True:
            message = await receive()
            consumed += len(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def exercise():
        messages = iter(
            [
                {"type": "http.request", "body": b"a" * 20, "more_body": True},
                {"type": "http.request", "body": b"b" * 20, "more_body": True},
                {"type": "http.request", "body": b"c" * 20, "more_body": False},
            ]
        )
        sent = []

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        middleware = UploadBoundaryMiddleware(downstream)
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/documents/upload",
                "headers": [(b"x-admin-token", b"test-admin-token-000000000000000002")],
            },
            receive,
            send,
        )
        return sent

    sent = asyncio.run(exercise())
    assert next(message["status"] for message in sent if message["type"] == "http.response.start") == 413
    assert consumed == 20


def test_declared_oversized_body_is_rejected_without_receive(monkeypatch) -> None:
    monkeypatch.setattr("app.core.upload_boundary.DOCUMENT_BODY_LIMIT", 32)

    async def downstream(_scope, _receive, _send):
        raise AssertionError("downstream must not run")

    async def exercise():
        sent = []

        async def receive():
            raise AssertionError("body must not be read")

        async def send(message):
            sent.append(message)

        await UploadBoundaryMiddleware(downstream)(
            {
                "type": "http",
                "method": "POST",
                "path": "/documents/upload",
                "headers": [
                    (b"x-admin-token", b"test-admin-token-000000000000000002"),
                    (b"content-length", b"33"),
                ],
            },
            receive,
            send,
        )
        return sent

    sent = asyncio.run(exercise())
    assert next(message["status"] for message in sent if message["type"] == "http.response.start") == 413


def test_chat_aggregate_limit_is_checked_before_parsing(monkeypatch) -> None:
    settings = SimpleNamespace(
        internal_token=SecretStr("test-internal-token-0000000000000001"),
        chat_media_max_bytes=8,
    )
    monkeypatch.setattr("app.core.upload_boundary.get_settings", lambda: settings)
    monkeypatch.setattr("app.core.upload_boundary.CHAT_BODY_OVERHEAD", 0)

    with patch.object(UploadFile, "write", wraps=UploadFile.write) as write:
        response = TestClient(app).post(
            "/chats/00000000-0000-0000-0000-000000000000/messages",
            headers={
                "X-Internal-Token": "test-internal-token-0000000000000001",
                "X-User-ID": "employee",
            },
            data={"content": "hello"},
            files={"media": ("x.png", b"x" * 20, "image/png")},
        )

    assert response.status_code == 413
    write.assert_not_called()
