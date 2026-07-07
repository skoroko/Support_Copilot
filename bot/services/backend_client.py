from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

import httpx


class BackendClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds, read=timeout_seconds),
        )
        self._chat_cache: dict[tuple[str, str], UUID] = {}

    async def get_or_create_chat(self, owner_external_id: str, interface: str) -> UUID:
        key = (owner_external_id, interface)
        if key in self._chat_cache:
            return self._chat_cache[key]

        response = await self._client.post(
            "/chats",
            json={"owner_external_id": owner_external_id, "interface": interface},
        )
        response.raise_for_status()
        chat_id = UUID(str(response.json()["chat_id"]))
        self._chat_cache[key] = chat_id
        return chat_id

    async def send_message(self, chat_id: UUID, content: str) -> AsyncIterator[str]:
        async with self._client.stream(
            "POST",
            f"/chats/{chat_id}/messages",
            json={"content": content},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if payload == "[DONE]":
                    break
                if payload:
                    yield payload

    async def clear_messages(self, chat_id: UUID) -> None:
        response = await self._client.delete(f"/chats/{chat_id}/messages")
        response.raise_for_status()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
