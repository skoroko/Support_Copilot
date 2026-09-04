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
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=DEFAULT_TIMEOUT)
        self._max_connect_attempts = max(1, max_connect_attempts)
        self._chat_cache: dict[tuple[str, str], UUID] = {}

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
                        elif event_type == "done":
                            return
                    return
            except NETWORK_ERRORS:
                if yielded_data or attempt >= self._max_connect_attempts:
                    raise
                await asyncio.sleep(0.2 * (2 ** (attempt - 1)))

    async def clear_messages(self, chat_id: UUID | str) -> None:
        response = await self._request_with_connect_retry("DELETE", f"/chats/{chat_id}/messages")
        response.raise_for_status()

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
