from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pydantic import SecretStr

from app.chat.domain import Chat, ChatMessage
from app.chat.service import MODERATION_OUTPUT_REPLACEMENT, ChatService, ModerationBlockedError
from app.core.config import Settings
from app.moderation.models import ModerationResult


class MemoryRepository:
    def __init__(self) -> None:
        self.chats: dict[UUID, Chat] = {}
        self.messages: dict[UUID, list[ChatMessage]] = {}
        self.moderation: list[tuple[str, ModerationResult]] = []

    async def create_chat(self, owner_external_id: str, interface: str, system_prompt: str | None = None) -> Chat:
        chat = Chat(owner_external_id=owner_external_id, interface=interface, system_prompt=system_prompt)
        self.chats[chat.id] = chat
        self.messages[chat.id] = []
        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        return self.chats.get(chat_id)

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        self.messages[chat_id].append(message)
        return message

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        return self.messages.get(chat_id, [])[-limit:]

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        self.messages[chat_id] = []

    async def record_moderation(self, chat_id: UUID, direction: str, result: ModerationResult) -> None:
        self.moderation.append((direction, result))


class FakeStream:
    def __init__(self, text: str) -> None:
        self.text = text
        self.sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.sent:
            raise StopAsyncIteration
        self.sent = True
        return {"choices": [{"delta": {"content": self.text}}]}


class FakeCompletions:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls = 0

    async def create(self, **_kwargs: Any) -> FakeStream:
        self.calls += 1
        return FakeStream(self.output)


class FakeLLM:
    def __init__(self, output: str) -> None:
        self.completions = FakeCompletions(output)
        self.chat = type("Chat", (), {"completions": self.completions})()


def settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key=SecretStr("sk-test"),
        chat_storage_dir=tmp_path,
        moderation_openai_enabled=False,
    )


@pytest.mark.asyncio
async def test_input_moderation_blocks_before_llm(tmp_path: Path) -> None:
    repo = MemoryRepository()
    llm = FakeLLM("answer")
    service = ChatService(repo, llm, settings(tmp_path))  # type: ignore[arg-type]
    chat = await service.create_chat("owner", "telegram")

    with pytest.raises(ModerationBlockedError):
        _ = [chunk async for chunk in service.send_message(chat.id, "kill yourself")]

    assert llm.completions.calls == 0
    assert repo.messages[chat.id] == []
    assert repo.moderation[-1][1].allowed is False


@pytest.mark.asyncio
async def test_output_moderation_replaces_before_yield(tmp_path: Path) -> None:
    class OutputBlockingModeration:
        async def check_input(self, _content: str) -> ModerationResult:
            return ModerationResult()

        async def check_output(self, _content: str) -> ModerationResult:
            return ModerationResult(
                allowed=False,
                categories=["violence"],
                reasons=["blocked"],
                blocked_by="keyword",
            )

    repo = MemoryRepository()
    service = ChatService(
        repo,
        FakeLLM("unsafe output"),
        settings(tmp_path),
        moderation_service=OutputBlockingModeration(),  # type: ignore[arg-type]
    )
    chat = await service.create_chat("owner", "telegram")

    chunks = [chunk async for chunk in service.send_message(chat.id, "normal question")]

    assert chunks == [MODERATION_OUTPUT_REPLACEMENT]
    assert repo.messages[chat.id][-1].content == MODERATION_OUTPUT_REPLACEMENT
    assert repo.messages[chat.id][-1].latency_ms is not None
