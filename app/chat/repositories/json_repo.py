from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import aiofiles

from app.chat.domain import Chat, ChatMessage


class JsonChatRepository:
    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        chat = Chat(
            owner_external_id=owner_external_id,
            interface=interface,
            system_prompt=system_prompt,
        )
        chat_dir = self._chat_dir(chat.id)
        chat_dir.mkdir(parents=True, exist_ok=False)
        async with aiofiles.open(self._chat_file(chat.id), "w", encoding="utf-8") as file:
            await file.write(chat.model_dump_json())
        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        path = self._chat_file(chat_id)
        if not path.exists():
            return None
        async with aiofiles.open(path, encoding="utf-8") as file:
            return Chat.model_validate_json(await file.read())

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        message = message.model_copy(update={"chat_id": chat_id})
        chat_dir = self._chat_dir(chat_id)
        if not chat_dir.exists():
            raise FileNotFoundError(f"Chat {chat_id} does not exist")
        async with aiofiles.open(self._messages_file(chat_id), "a", encoding="utf-8") as file:
            await file.write(message.model_dump_json())
            await file.write("\n")
        return message

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        if limit <= 0:
            return []
        path = self._messages_file(chat_id)
        if not path.exists():
            return []
        async with aiofiles.open(path, encoding="utf-8") as file:
            lines = await file.readlines()

        messages: list[ChatMessage] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict) and payload.get("type") == "soft_delete":
                messages = []
                continue
            messages.append(ChatMessage.model_validate(payload))
        return messages[-limit:]

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        chat_dir = self._chat_dir(chat_id)
        if not chat_dir.exists():
            return
        marker = {"type": "soft_delete", "at": datetime.now(timezone.utc).isoformat()}
        async with aiofiles.open(self._messages_file(chat_id), "a", encoding="utf-8") as file:
            await file.write(json.dumps(marker, ensure_ascii=False, separators=(",", ":")))
            await file.write("\n")

    def _chat_dir(self, chat_id: UUID) -> Path:
        return self.base_dir / "chats" / str(chat_id)

    def _chat_file(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "chat.json"

    def _messages_file(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "messages.jsonl"
