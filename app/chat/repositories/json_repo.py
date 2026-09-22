from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import aiofiles
from filelock import AsyncFileLock

from app.admin.models import BroadcastItem, StatsOut, UserOut
from app.chat.domain import Chat, ChatMessage
from app.chat.feedback import MessageFeedback
from app.moderation.models import ModerationResult


class JsonChatRepository:
    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self._lock = AsyncFileLock(self.base_dir / ".chat-repository.lock", timeout=10)

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
        async with self._lock:
            chat_dir = self._chat_dir(chat.id)
            chat_dir.mkdir(parents=True, exist_ok=False)
            await self._atomic_write_text(self._chat_file(chat.id), chat.model_dump_json())
        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        async with self._lock:
            path = self._chat_file(chat_id)
            if not path.exists():
                return None
            async with aiofiles.open(path, encoding="utf-8") as file:
                return Chat.model_validate_json(await file.read())

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        message = message.model_copy(update={"chat_id": chat_id})
        async with self._lock:
            self._require_chat(chat_id)
            await self._append_text(self._messages_file(chat_id), message.model_dump_json() + "\n")
        return message

    async def append_turn(
        self,
        chat_id: UUID,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
    ) -> tuple[ChatMessage, ChatMessage]:
        messages = (
            user_message.model_copy(update={"chat_id": chat_id}),
            assistant_message.model_copy(update={"chat_id": chat_id}),
        )
        payload = "".join(message.model_dump_json() + "\n" for message in messages)
        async with self._lock:
            self._require_chat(chat_id)
            await self._append_text(self._messages_file(chat_id), payload)
        return messages

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        if limit <= 0:
            return []
        async with self._lock:
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

    async def get_message(self, chat_id: UUID, message_id: UUID) -> ChatMessage | None:
        messages = await self.list_messages(chat_id, limit=1_000_000)
        return next((message for message in messages if message.id == message_id), None)

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        marker = {"type": "soft_delete", "at": datetime.now(timezone.utc).isoformat()}
        async with self._lock:
            if not self._chat_dir(chat_id).exists():
                return
            payload = json.dumps(marker, ensure_ascii=False, separators=(",", ":")) + "\n"
            await self._append_text(self._messages_file(chat_id), payload)

    async def save_feedback(self, feedback: MessageFeedback) -> MessageFeedback:
        async with self._lock:
            rows = await self._read_jsonl_unlocked(self._feedback_file())
            unique_rows: list[dict] = []
            replaced = False
            for row in rows:
                if row.get("owner_external_id") == feedback.owner_external_id and row.get("message_id") == str(
                    feedback.message_id
                ):
                    if not replaced:
                        unique_rows.append(feedback.model_dump(mode="json"))
                        replaced = True
                    continue
                unique_rows.append(row)
            if not replaced:
                unique_rows.append(feedback.model_dump(mode="json"))
            await self._write_jsonl_unlocked(self._feedback_file(), unique_rows)
        return feedback

    async def record_moderation(
        self,
        chat_id: UUID,
        direction: str,
        result: ModerationResult,
    ) -> None:
        async with self._lock:
            rows = await self._read_jsonl_unlocked(self._moderation_file())
            rows.append(
                {
                    "chat_id": str(chat_id),
                    "direction": direction,
                    "allowed": result.allowed,
                    "categories": result.categories,
                    "blocked_by": result.blocked_by,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            await self._write_jsonl_unlocked(self._moderation_file(), rows)

    async def get_admin_stats(self) -> StatsOut:
        since = datetime.now(timezone.utc).timestamp() - 24 * 60 * 60
        total_messages = 0
        active_users: set[str] = set()
        latencies: list[float] = []
        for chat in await self._load_chats():
            recent = [
                message
                for message in await self.list_messages(chat.id, limit=1_000_000)
                if message.created_at.timestamp() >= since
            ]
            total_messages += len(recent)
            if recent:
                active_users.add(chat.owner_external_id)
            latencies.extend(message.latency_ms for message in recent if message.latency_ms is not None)

        moderation_rows = [
            row
            for row in await self._read_jsonl(self._moderation_file())
            if self._row_timestamp(row) >= since
        ]
        feedback_rows = [
            row for row in await self._read_jsonl(self._feedback_file()) if self._row_timestamp(row) >= since
        ]
        blocked = sum(1 for row in moderation_rows if not bool(row.get("allowed", True)))
        feedback_up = sum(1 for row in feedback_rows if row.get("value") == "up")
        return StatsOut(
            total_messages=total_messages,
            active_users=len(active_users),
            avg_latency_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            moderation_block_rate=round(blocked / len(moderation_rows), 4) if moderation_rows else 0.0,
            feedback_up_ratio=round(feedback_up / len(feedback_rows), 4) if feedback_rows else 0.0,
        )

    async def list_admin_users(self, limit: int = 50, interface: str | None = None) -> list[UserOut]:
        aggregated: dict[str, dict[str, object]] = {}
        for chat in await self._load_chats():
            if interface is not None and chat.interface != interface:
                continue
            messages = await self.list_messages(chat.id, limit=1_000_000)
            last_seen = max((message.created_at for message in messages), default=chat.created_at)
            current = aggregated.setdefault(
                chat.owner_external_id,
                {"chat_count": 0, "last_seen_at": last_seen},
            )
            current["chat_count"] = int(current["chat_count"]) + 1
            current["last_seen_at"] = max(current["last_seen_at"], last_seen)  # type: ignore[type-var]
        users = [
            UserOut(
                owner_external_id=owner,
                chat_count=int(values["chat_count"]),
                last_seen_at=values["last_seen_at"],  # type: ignore[arg-type]
            )
            for owner, values in aggregated.items()
        ]
        users.sort(key=lambda user: user.last_seen_at, reverse=True)
        return users[: max(0, limit)]

    async def enqueue_broadcast(self, message: str, interface: str) -> BroadcastItem:
        item = BroadcastItem(message=message, interface=interface)
        async with self._lock:
            rows = await self._read_jsonl_unlocked(self._broadcast_file())
            rows.append(item.model_dump(mode="json"))
            await self._write_jsonl_unlocked(self._broadcast_file(), rows)
        return item

    async def list_pending_broadcasts(self, limit: int = 10) -> list[BroadcastItem]:
        items = [
            BroadcastItem.model_validate(row)
            for row in await self._read_jsonl(self._broadcast_file())
            if row.get("status") == "pending"
        ]
        items.sort(key=lambda item: item.created_at)
        return items[: max(0, limit)]

    async def update_broadcast_status(self, broadcast_id: UUID, status: str) -> BroadcastItem | None:
        async with self._lock:
            rows = await self._read_jsonl_unlocked(self._broadcast_file())
            updated: BroadcastItem | None = None
            output: list[dict] = []
            for row in rows:
                item = BroadcastItem.model_validate(row)
                if item.id == broadcast_id:
                    item = item.model_copy(update={"status": status, "updated_at": datetime.now(timezone.utc)})
                    updated = item
                output.append(item.model_dump(mode="json"))
            if updated is not None:
                await self._write_jsonl_unlocked(self._broadcast_file(), output)
        return updated

    async def _load_chats(self) -> list[Chat]:
        async with self._lock:
            chats_dir = self.base_dir / "chats"
            if not chats_dir.exists():
                return []
            chats: list[Chat] = []
            for path in chats_dir.glob("*/chat.json"):
                try:
                    chats.append(Chat.model_validate_json(path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
            return chats

    @staticmethod
    def _row_timestamp(row: dict) -> float:
        try:
            return datetime.fromisoformat(str(row.get("created_at"))).timestamp()
        except (TypeError, ValueError):
            return 0.0

    async def _read_jsonl(self, path: Path) -> list[dict]:
        async with self._lock:
            return await self._read_jsonl_unlocked(path)

    async def _read_jsonl_unlocked(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        async with aiofiles.open(path, encoding="utf-8") as file:
            return [json.loads(line) for line in await file.readlines() if line.strip()]

    async def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        async with self._lock:
            await self._write_jsonl_unlocked(path, rows)

    async def _write_jsonl_unlocked(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
            for row in rows
        )
        await self._atomic_write_text(path, payload)

    async def _append_text(self, path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "a", encoding="utf-8") as file:
            await file.write(payload)
            await file.flush()
            await asyncio.to_thread(os.fsync, file.fileno())

    async def _atomic_write_text(self, path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            async with aiofiles.open(temporary, "w", encoding="utf-8") as file:
                await file.write(payload)
                await file.flush()
                await asyncio.to_thread(os.fsync, file.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _require_chat(self, chat_id: UUID) -> None:
        if not self._chat_dir(chat_id).exists():
            raise FileNotFoundError(f"Chat {chat_id} does not exist")

    def _chat_dir(self, chat_id: UUID) -> Path:
        return self.base_dir / "chats" / str(chat_id)

    def _chat_file(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "chat.json"

    def _messages_file(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "messages.jsonl"

    def _feedback_file(self) -> Path:
        return self.base_dir / "message_feedback.jsonl"

    def _moderation_file(self) -> Path:
        return self.base_dir / "moderation_events.jsonl"

    def _broadcast_file(self) -> Path:
        return self.base_dir / "broadcast_queue.jsonl"
