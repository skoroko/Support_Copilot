from __future__ import annotations

import re
import secrets
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import get_settings


DOCUMENT_BODY_LIMIT = 51 * 1024 * 1024  # 50 MiB file plus multipart framing
CHAT_MEDIA_MAX_FILES = 4
CHAT_BODY_OVERHEAD = 1024 * 1024
_CHAT_MESSAGES = re.compile(r"^/chats/[^/]+/messages$")


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


class UploadBoundaryMiddleware:
    """Authenticate and bound upload streams before FastAPI parses multipart."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        document = path == "/documents/upload"
        chat = bool(_CHAT_MESSAGES.fullmatch(path))
        if not document and not chat:
            await self.app(scope, receive, send)
            return

        settings = get_settings()
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        if document:
            expected = settings.admin_token.get_secret_value().encode()
            supplied = headers.get(b"x-admin-token", b"")
            if not supplied or not secrets.compare_digest(supplied, expected):
                await _error(401, "admin_unauthorized", "Valid X-Admin-Token is required.")(scope, receive, send)
                return
            limit = DOCUMENT_BODY_LIMIT
        else:
            expected = settings.internal_token.get_secret_value().encode()
            supplied = headers.get(b"x-internal-token", b"")
            user_id = headers.get(b"x-user-id", b"").decode("utf-8", errors="replace").strip()
            interface = headers.get(b"x-user-interface", b"service").decode("utf-8", errors="replace").strip().casefold()
            if not supplied or not secrets.compare_digest(supplied, expected):
                response = _error(401, "service_unauthorized", "Valid service authentication is required.")
                await response(scope, receive, send)
                return
            if not user_id or len(user_id) > 200 or any(ord(char) < 32 for char in user_id):
                response = _error(401, "principal_required", "A valid authenticated principal is required.")
                await response(scope, receive, send)
                return
            if interface not in {"telegram", "cli", "web", "service"}:
                response = _error(401, "principal_interface_invalid", "A valid principal interface is required.")
                await response(scope, receive, send)
                return
            limit = settings.chat_media_max_bytes * 2 + CHAT_BODY_OVERHEAD

        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = -1
            if declared < 0 or declared > limit:
                await _error(413, "request_body_too_large", "Request body exceeds the upload limit.")(
                    scope, receive, send
                )
                return

        consumed = 0
        response_started = False

        class BodyTooLarge(Exception):
            pass

        async def bounded_receive() -> Message:
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    raise BodyTooLarge
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, tracked_send)
        except BodyTooLarge:
            if not response_started:
                await _error(413, "request_body_too_large", "Request body exceeds the upload limit.")(
                    scope, receive, send
                )
            else:
                raise
