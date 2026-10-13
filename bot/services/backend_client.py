from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from uuid import UUID

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=5.0)
STREAM_TIMEOUT = httpx.Timeout(connect=3.0, read=120.0, write=10.0, pool=5.0)
NETWORK_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


class BackendStreamError(httpx.HTTPError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class StreamReplace:
    """Canonical answer text replacing the accumulated streamed draft.

    The backend emits it after grounding/moderation when the final text can
    differ from the token deltas already shown.
    """

    text: str


class BackendClient:
    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.AsyncClient | None = None,
        max_connect_attempts: int = 3,
        internal_token: str,
        admin_token: str | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=DEFAULT_TIMEOUT)
        self._max_connect_attempts = max(1, max_connect_attempts)
        self._chat_cache: dict[tuple[str, str], UUID] = {}
        self._last_message_ids: dict[str, UUID] = {}
        self._last_sources: dict[str, list[dict]] = {}
        self._last_notices: dict[str, str] = {}
        self._last_approvals: dict[str, dict | None] = {}
        self._chat_owners: dict[str, str] = {}
        self._internal_token = internal_token
        self._admin_headers = {"X-Admin-Token": admin_token} if admin_token else {}

    async def get_or_create_chat(self, owner_external_id: str, interface: str) -> UUID:
        key = (owner_external_id, interface)
        if key in self._chat_cache:
            return self._chat_cache[key]
        response = await self._request_with_connect_retry(
            "POST",
            "/chats",
            headers=self._principal_headers(owner_external_id, interface=interface),
        )
        response.raise_for_status()
        chat_id = UUID(str(response.json()["chat_id"]))
        self._chat_cache[key] = chat_id
        self._chat_owners[str(chat_id)] = owner_external_id
        return chat_id

    async def send_message(
        self,
        chat_id: UUID | str,
        content: str,
        media: Sequence[tuple[bytes, str]] | None = None,
    ) -> AsyncIterator[str | StreamReplace]:
        files = None
        if media:
            files = [
                ("media", (self._filename_for_mime(mime), data, mime or "application/octet-stream"))
                for data, mime in media
            ]

        yielded_data = False
        chat_key = str(chat_id)
        self._last_sources[chat_key] = []
        self._last_notices[chat_key] = ""
        self._last_approvals[chat_key] = None
        for attempt in range(1, self._max_connect_attempts + 1):
            try:
                async with self._client.stream(
                    "POST",
                    f"/chats/{chat_id}/messages",
                    data={"content": content},
                    files=files,
                    headers=self._principal_headers_for_chat(chat_id),
                    timeout=STREAM_TIMEOUT,
                ) as response:
                    response.raise_for_status()
                    event_name = "message"
                    async for line in response.aiter_lines():
                        if not line:
                            event_name = "message"
                            continue
                        if line.startswith("event:"):
                            event_name = line.removeprefix("event:").strip() or "message"
                            continue
                        if not line.startswith("data:"):
                            continue
                        raw = line.removeprefix("data:").strip()
                        if not raw:
                            continue
                        payload = json.loads(raw)
                        if event_name == "sources":
                            self._last_sources[chat_key] = self._sources_from_payload(payload)
                            continue
                        if event_name == "error" or payload.get("type") == "error":
                            raise BackendStreamError(
                                str(payload.get("code") or "chat_stream_failed"),
                                str(payload.get("message") or "Backend stream failed"),
                            )
                        event_type = payload.get("type")
                        if event_type == "token":
                            yielded_data = True
                            yield str(payload.get("delta", ""))
                        elif event_type == "replace":
                            yielded_data = True
                            yield StreamReplace(str(payload.get("text", "")))
                        elif event_type == "message":
                            message_id = payload.get("message_id")
                            if message_id:
                                self._last_message_ids[str(chat_id)] = UUID(str(message_id))
                        elif event_type == "notice":
                            notice = str(payload.get("message") or "").strip()
                            if notice:
                                self._last_notices[chat_key] = notice
                        elif event_type == "approval":
                            self._last_approvals[chat_key] = payload.get("approval")
                        elif event_type == "done":
                            message_id = payload.get("message_id")
                            if message_id:
                                self._last_message_ids[str(chat_id)] = UUID(str(message_id))
                            # Keep reading: the server terminates the stream with
                            # the named ``sources`` event after this legacy marker.
                            continue
                    return
            except NETWORK_ERRORS:
                if yielded_data or attempt >= self._max_connect_attempts:
                    raise
                await asyncio.sleep(0.2 * (2 ** (attempt - 1)))

    async def clear_messages(self, chat_id: UUID | str) -> None:
        response = await self._request_with_connect_retry(
            "DELETE",
            f"/chats/{chat_id}/messages",
            headers=self._principal_headers_for_chat(chat_id),
        )
        response.raise_for_status()
        self._last_approvals[str(chat_id)] = None

    async def list_messages(self, chat_id: UUID | str, limit: int = 10) -> list[dict]:
        response = await self._request_with_connect_retry("GET", f"/chats/{chat_id}/messages",
                                                          params={"limit": limit},
                                                          headers=self._principal_headers_for_chat(chat_id))
        response.raise_for_status()
        return list(response.json())

    def last_response_message_id(self, chat_id: UUID | str) -> UUID | None:
        return self._last_message_ids.get(str(chat_id))

    def last_response_sources(self, chat_id: UUID | str) -> list[dict]:
        return [dict(source) for source in self._last_sources.get(str(chat_id), [])]

    def last_response_notice(self, chat_id: UUID | str) -> str:
        notice = self._last_notices.get(str(chat_id), "")
        draft = self.last_approval(chat_id)
        if draft:
            notice += f"\n\nПодтвердите отправку.\nПолучатель (chat_id): {draft['chat_id']}\nТочный текст:\n{draft['text']}"
        return notice.strip()

    def last_approval(self, chat_id: UUID | str) -> dict | None:
        return self._last_approvals.get(str(chat_id))

    async def pending_approval(self, chat_id: UUID | str) -> dict | None:
        response = await self._request_with_connect_retry("GET", f"/chats/{chat_id}/approval",
                                                          headers=self._principal_headers_for_chat(chat_id))
        response.raise_for_status()
        draft = response.json().get("approval")
        self._last_approvals[str(chat_id)] = draft
        return draft

    async def resume_approval(self, chat_id: UUID | str, approval_id: UUID | str, approved: bool,
                              *, owner_external_id: str) -> dict:
        # Never retry an approval mutation. A lost response must not turn a
        # repeated callback into approval of a different pending operation.
        response = await self._client.post(f"/chats/{chat_id}/approvals/{approval_id}",
                                          json={"approved": approved},
                                          headers=self._principal_headers_for_chat(chat_id, owner_external_id),
                                          timeout=STREAM_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        self._last_approvals[str(chat_id)] = result.get("approval")
        self._last_sources[str(chat_id)] = self._sources_from_payload(result)
        if result.get("message_id"):
            self._last_message_ids[str(chat_id)] = UUID(str(result["message_id"]))
        return result

    async def get_message_sources(
        self,
        chat_id: UUID | str,
        message_id: UUID | str,
        *,
        owner_external_id: str | None = None,
    ) -> list[dict]:
        response = await self._request_with_connect_retry(
            "GET",
            f"/chats/{chat_id}/messages/{message_id}/sources",
            headers=self._principal_headers_for_chat(chat_id, owner_external_id),
        )
        response.raise_for_status()
        return self._sources_from_payload(response.json())

    async def save_feedback(
        self,
        chat_id: UUID | str,
        message_id: UUID | str,
        value: str,
        shown_sources: list[dict] | None = None,
        owner_external_id: str | None = None,
    ) -> dict:
        sources = shown_sources
        if sources is None:
            sources = self.last_response_sources(chat_id) or await self.get_message_sources(
                chat_id,
                message_id,
                owner_external_id=owner_external_id,
            )
        response = await self._request_with_connect_retry(
            "POST",
            f"/chats/{chat_id}/messages/{message_id}/feedback",
            json={"value": value, "shown_sources": sources},
            headers=self._principal_headers_for_chat(chat_id, owner_external_id),
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

    async def claim_broadcasts(self, *, limit: int = 10, lease_seconds: int = 300) -> list[dict]:
        response = await self._request_with_connect_retry(
            "POST",
            "/chats/admin/broadcasts/claim",
            json={"limit": limit, "lease_seconds": lease_seconds},
            headers=self._admin_headers,
        )
        response.raise_for_status()
        return list(response.json())

    async def update_broadcast_recipient(
        self, broadcast_id: UUID | str, lease_id: UUID | str,
        chat_id: int, attempt_id: UUID | str, status: str,
    ) -> dict:
        response = await self._request_with_connect_retry(
            "POST",
            f"/chats/admin/broadcasts/{broadcast_id}/recipients/{chat_id}/state",
            json={"lease_id": str(lease_id), "attempt_id": str(attempt_id), "status": status},
            headers=self._admin_headers,
        )
        response.raise_for_status()
        return response.json()

    async def finish_broadcast(self, broadcast_id: UUID | str, lease_id: UUID | str) -> dict:
        response = await self._request_with_connect_retry(
            "POST",
            f"/chats/admin/broadcasts/{broadcast_id}/finish",
            json={"lease_id": str(lease_id)},
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

    def _principal_headers(self, owner_external_id: str, *, interface: str = "telegram") -> dict[str, str]:
        return {
            "X-Internal-Token": self._internal_token,
            "X-User-ID": owner_external_id,
            "X-User-Interface": interface,
        }

    def _principal_headers_for_chat(
        self,
        chat_id: UUID | str,
        owner_external_id: str | None = None,
    ) -> dict[str, str]:
        owner = owner_external_id or self._chat_owners.get(str(chat_id))
        if not owner:
            raise ValueError("Authenticated chat owner is required")
        # Only get_or_create_chat may bind a chat to an owner: a caller-supplied
        # principal (e.g. a feedback click in a group) must not remap it.
        return self._principal_headers(owner)

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

    @staticmethod
    def _sources_from_payload(payload: object) -> list[dict]:
        values = payload.get("sources") if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            return []
        return [dict(source) for source in values if isinstance(source, dict)]
