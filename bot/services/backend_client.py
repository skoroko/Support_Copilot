from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from uuid import UUID

import httpx


DEFAULT_TIMEOUT = httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=5.0)
STREAM_TIMEOUT = httpx.Timeout(connect=3.0, read=120.0, write=10.0, pool=5.0)
NETWORK_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


class BackendClient:
    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.AsyncClient | None = None,
        max_connect_attempts: int = 3,
        admin_token: str | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=DEFAULT_TIMEOUT)
        self._max_connect_attempts = max(1, max_connect_attempts)
        self._chat_cache: dict[tuple[str, str], UUID] = {}
        self._last_message_ids: dict[str, UUID] = {}
        self._admin_headers = {"X-Admin-Token": admin_token} if admin_token else {}

    async def get_or_create_chat(self, owner_external_id: str, interface: str) -> UUID:
        key = (owner_external_id, interface)
        if key in self._chat_cache:
            return self._chat_cache[key]
        response = await self._request_with_connect_retry(
            "POST", "/chats", json={"owner_external_id": owner_external_id, "interface": interface}
        )
        response.raise_for_status()
        chat_id = UUID(str(response.json()["chat_id"]))
        self._chat_cache[key] = chat_id
        return chat_id

    async def send_message(
        self,
        chat_id: UUID | str,
        content: str,
        media: bytes | None = None,
        mime: str | None = None,
    ) -> AsyncIterator[str]:
        files = None
        if media is not None:
            files = {"media": (self._filename_for_mime(mime), media, mime or "application/octet-stream")}

        yielded_data = False
        for attempt in range(1, self._max_connect_attempts + 1):
            try:
                async with self._client.stream(
                    "POST",
                    f"/chats/{chat_id}/messages",
                    data={"content": content},
                    files=files,
                    timeout=STREAM_TIMEOUT,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line.removeprefix("data:").strip()
                        if not raw:
                            continue
                        payload = json.loads(raw)
                        event_type = payload.get("type")
                        if event_type == "token":
                            yielded_data = True
                            yield str(payload.get("delta", ""))
                        elif event_type == "message":
                            message_id = payload.get("message_id")
                            if message_id:
                                self._last_message_ids[str(chat_id)] = UUID(str(message_id))
                        elif event_type == "done":
                            message_id = payload.get("message_id")
                            if message_id:
                                self._last_message_ids[str(chat_id)] = UUID(str(message_id))
                            return
                    return
            except NETWORK_ERRORS:
                if yielded_data or attempt >= self._max_connect_attempts:
                    raise
                await asyncio.sleep(0.2 * (2 ** (attempt - 1)))

    async def clear_messages(self, chat_id: UUID | str) -> None:
        response = await self._request_with_connect_retry("DELETE", f"/chats/{chat_id}/messages")
        response.raise_for_status()

    def last_response_message_id(self, chat_id: UUID | str) -> UUID | None:
        return self._last_message_ids.get(str(chat_id))

    async def save_feedback(self, chat_id: UUID | str, message_id: UUID | str, value: str) -> dict:
        response = await self._request_with_connect_retry(
            "POST",
            f"/chats/{chat_id}/messages/{message_id}/feedback",
            json={"value": value},
        )
        response.raise_for_status()
        return response.json()

    async def get_admin_stats(self) -> dict:
        response = await self._request_with_connect_retry(
            "GET",
            "/chats/admin/stats",
            headers=self._admin_headers,
        )
        response.raise_for_status()
        return response.json()

    async def get_admin_users(self, *, limit: int = 50, interface: str | None = None) -> list[dict]:
        params: dict[str, str | int] = {"limit": limit}
        if interface is not None:
            params["interface"] = interface
        response = await self._request_with_connect_retry(
            "GET",
            "/chats/admin/users",
            params=params,
            headers=self._admin_headers,
        )
        response.raise_for_status()
        return list(response.json())

    async def create_broadcast(self, message: str, interface: str = "telegram") -> dict:
        response = await self._request_with_connect_retry(
            "POST",
            "/chats/admin/broadcast",
            json={"message": message, "interface_filter": interface},
            headers=self._admin_headers,
        )
        response.raise_for_status()
        return response.json()

    async def get_pending_broadcasts(self, *, limit: int = 10) -> list[dict]:
        response = await self._request_with_connect_retry(
            "GET",
            "/chats/admin/broadcasts/pending",
            params={"limit": limit},
            headers=self._admin_headers,
        )
        response.raise_for_status()
        return list(response.json())

    async def update_broadcast_status(self, broadcast_id: UUID | str, status: str) -> dict:
        response = await self._request_with_connect_retry(
            "POST",
            f"/chats/admin/broadcasts/{broadcast_id}/status",
            json={"status": status},
            headers=self._admin_headers,
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request_with_connect_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        for attempt in range(1, self._max_connect_attempts + 1):
            try:
                return await self._client.request(method, url, **kwargs)
            except NETWORK_ERRORS:
                if attempt >= self._max_connect_attempts:
                    raise
                await asyncio.sleep(0.2 * (2 ** (attempt - 1)))
        raise RuntimeError("unreachable")

    @staticmethod
    def _filename_for_mime(mime: str | None) -> str:
        extensions = {
            "image/jpeg": "image.jpg",
            "image/png": "image.png",
            "image/webp": "image.webp",
            "audio/ogg": "audio.ogg",
            "application/ogg": "audio.ogg",
            "audio/mpeg": "audio.mp3",
            "audio/mp4": "audio.m4a",
            "application/pdf": "document.pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document.docx",
        }
        return extensions.get((mime or "").lower(), "file.bin")
