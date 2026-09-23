from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pydantic import SecretStr

from app.chat.domain import Chat, ChatMessage
from app.chat.repository import ChatRepository
from app.chat.service import (
    ChatNotFoundError,
    ChatService,
    MEDIA_IMAGE_NO_TEXT_REPLY,
    RAG_MEDIA_TEXT_MAX_CHARS,
    count_tokens,
    fit_to_budget,
)
from app.core.config import Settings


class InMemoryRepository:
    def __init__(self) -> None:
        self.chats: dict[UUID, Chat] = {}
        self.messages: dict[UUID, list[ChatMessage]] = {}

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        chat = Chat(owner_external_id=owner_external_id, interface=interface, system_prompt=system_prompt)
        self.chats[chat.id] = chat
        self.messages[chat.id] = []
        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        return self.chats.get(chat_id)

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        self.messages.setdefault(chat_id, []).append(message.model_copy(update={"chat_id": chat_id}))
        return message

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        return self.messages.get(chat_id, [])[-limit:]

    async def get_message(self, chat_id: UUID, message_id: UUID) -> ChatMessage | None:
        return next((message for message in self.messages.get(chat_id, []) if message.id == message_id), None)

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        self.messages[chat_id] = []


class FakeStream:
    def __init__(self, parts: list[str]) -> None:
        self.parts = parts

    def __aiter__(self) -> "FakeStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self.parts:
            raise StopAsyncIteration
        return {"choices": [{"delta": {"content": self.parts.pop(0)}}]}


class FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> FakeStream:
        self.last_kwargs = kwargs
        return FakeStream(["Аня"])


class FakeLLM:
    def __init__(self) -> None:
        completions = FakeCompletions()
        self.chat = type("Chat", (), {"completions": completions})()
        self.completions = completions


class FailingStream:
    def __init__(self) -> None:
        self.sent = False

    def __aiter__(self) -> "FailingStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self.sent:
            self.sent = True
            return {"choices": [{"delta": {"content": "partial"}}]}
        raise RuntimeError("provider disconnected")


class FailingLLM:
    def __init__(self) -> None:
        async def create(**_kwargs: Any) -> FailingStream:
            return FailingStream()

        completions = type("Completions", (), {"create": staticmethod(create)})()
        self.chat = type("Chat", (), {"completions": completions})()


class FakeRAG:
    def __init__(self) -> None:
        self.question: str | None = None
        self.history: list[dict[str, Any]] | None = None

    def answer(self, question: str, *, history: list[dict[str, Any]]) -> dict[str, Any]:
        self.question = question
        self.history = history
        return {
            "answer": "Для электроники действует отдельный срок возврата [1].",
            "sources": [
                {
                    "id": "electronics-policy-1",
                    "file_name": "returns.md",
                    "page": 2,
                    "score": 0.91,
                    "snippet": "Электроника возвращается в течение 14 дней.",
                    "confident": True,
                }
            ],
        }


def build_settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key=SecretStr("sk-test"),
        default_model="gpt-5.4-mini",
        request_timeout=30,
        redis_url="redis://localhost:6379/0",
        cache_ttl_seconds=60,
        cors_origins=["*"],
        chat_repository="json",
        chat_storage_dir=tmp_path,
        chat_database_url="postgresql+asyncpg://support:support@localhost:5432/support_copilot",
        chat_context_strategy="sliding",
        chat_context_window=10,
        chat_model_context_tokens=128_000,
        chat_response_tokens=500,
        chat_safety_margin_tokens=500,
    )


def test_count_tokens_has_chatml_overhead() -> None:
    assert count_tokens([{"role": "user", "content": "hello"}]) > 2


def test_fit_to_budget_preserves_system_message() -> None:
    messages = [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "old " * 200},
        {"role": "assistant", "content": "recent"},
    ]

    fitted = fit_to_budget(messages, budget=count_tokens([messages[0], messages[2]]))

    assert fitted[0]["role"] == "system"
    assert fitted[-1]["content"] == "recent"
    assert all(message["content"] != messages[1]["content"] for message in fitted)


@pytest.mark.asyncio
async def test_send_message_builds_context_streams_and_persists_assistant(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    llm = FakeLLM()
    service = ChatService(repo, llm, build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram", "support system")
    await repo.append_message(chat.id, ChatMessage(chat_id=chat.id, role="user", content="Меня зовут Аня"))

    chunks = [chunk async for chunk in service.send_message(chat.id, "Как меня зовут?")]

    assert chunks == ["Аня"]
    assert llm.completions.last_kwargs["messages"][0] == {"role": "system", "content": "support system"}
    assert any(message["content"] == "Меня зовут Аня" for message in llm.completions.last_kwargs["messages"])
    assert any(message["content"] == "Как меня зовут?" for message in llm.completions.last_kwargs["messages"])
    assert llm.completions.last_kwargs["stream_options"] == {"include_usage": True}
    stored = await repo.list_messages(chat.id)
    assert [message.role for message in stored] == ["user", "user", "assistant"]
    assert stored[-1].content == "Аня"


@pytest.mark.asyncio
async def test_send_message_unknown_chat_raises(tmp_path: Path) -> None:
    service = ChatService(InMemoryRepository(), FakeLLM(), build_settings(tmp_path))

    with pytest.raises(ChatNotFoundError):
        [chunk async for chunk in service.send_message(UUID("00000000-0000-0000-0000-000000000000"), "hello")]


@pytest.mark.asyncio
async def test_provider_failure_emits_no_partial_text_and_persists_no_orphan_turn(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    service = ChatService(repo, FailingLLM(), build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")

    with pytest.raises(RuntimeError, match="provider disconnected"):
        _ = [chunk async for chunk in service.send_message(chat.id, "hello")]

    assert await repo.list_messages(chat.id) == []


@pytest.mark.asyncio
async def test_media_part_is_restored_in_followup_context(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    llm = FakeLLM()
    service = ChatService(repo, llm, build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")
    part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}}

    _ = [
        chunk
        async for chunk in service.send_message(
            chat.id,
            "Что на фото?",
            media_refs={"mime": "image/png", "size": 3, "filename": "x.png", "part": part},
        )
    ]

    user_message = next(message for message in llm.completions.last_kwargs["messages"] if message["role"] == "user")
    assert user_message["content"] == [{"type": "text", "text": "Что на фото?"}, part]


@pytest.mark.asyncio
async def test_rag_uses_prior_history_and_exposes_shown_sources(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    llm = FakeLLM()
    rag = FakeRAG()
    service = ChatService(repo, llm, build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")
    await repo.append_message(chat.id, ChatMessage(chat_id=chat.id, role="user", content="Каков срок возврата?"))
    await repo.append_message(
        chat.id,
        ChatMessage(chat_id=chat.id, role="assistant", content="Для большинства товаров — 14 дней."),
    )

    chunks = [
        chunk
        async for chunk in service.send_message(chat.id, "А для электроники?", rag_service=rag)
    ]

    assert chunks == ["Для электроники действует отдельный срок возврата [1]."]
    assert rag.question == "А для электроники?"
    assert [item["content"] for item in rag.history or []] == [
        "Каков срок возврата?",
        "Для большинства товаров — 14 дней.",
    ]
    assert llm.completions.last_kwargs is None
    assert service.last_response_sources(chat.id) == [
        {
            "id": "electronics-policy-1",
            "file_name": "returns.md",
            "page": 2,
            "score": 0.91,
            "snippet": "Электроника возвращается в течение 14 дней.",
            "confident": True,
        }
    ]
    stored = await repo.list_messages(chat.id)
    assert stored[-1].rag_sources == service.last_response_sources(chat.id)
    assert await service.response_sources(chat.id, stored[-1].id) == service.last_response_sources(chat.id)


@pytest.mark.asyncio
async def test_rag_question_merges_extracted_media_text(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    llm = FakeLLM()
    rag = FakeRAG()
    service = ChatService(repo, llm, build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")
    part = {"type": "text", "text": "[пользователь сказал голосом]:\nпочему модель не сходится"}

    chunks = [
        chunk
        async for chunk in service.send_message(
            chat.id,
            "[голосовое сообщение]",
            media_refs={"mime": "audio/ogg", "size": 10, "filename": "v.ogg", "part": part},
            rag_service=rag,
        )
    ]

    assert chunks == ["Для электроники действует отдельный срок возврата [1]."]
    assert rag.question == (
        "[голосовое сообщение]\n[пользователь сказал голосом]:\nпочему модель не сходится"
    )


@pytest.mark.asyncio
async def test_rag_question_bounds_extracted_document_text(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    rag = FakeRAG()
    service = ChatService(repo, FakeLLM(), build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")
    part = {"type": "text", "text": "[документ PDF]:\n" + "очень длинный текст " * 2000}

    _ = [
        chunk
        async for chunk in service.send_message(
            chat.id,
            "проверь документ",
            media_refs={"mime": "application/pdf", "size": 10, "filename": "d.pdf", "part": part},
            rag_service=rag,
        )
    ]

    assert rag.question is not None
    assert len(rag.question) == RAG_MEDIA_TEXT_MAX_CHARS
    assert rag.question.startswith("проверь документ\n[документ PDF]:")


@pytest.mark.asyncio
async def test_rag_image_without_caption_refuses_without_retrieval(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    llm = FakeLLM()
    rag = FakeRAG()
    service = ChatService(repo, llm, build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")
    part = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aW1n"}}

    chunks = [
        chunk
        async for chunk in service.send_message(
            chat.id,
            "[фото]",
            media_refs={"mime": "image/jpeg", "size": 3, "filename": "x.jpg", "part": part},
            rag_service=rag,
        )
    ]

    assert chunks == [MEDIA_IMAGE_NO_TEXT_REPLY]
    assert rag.question is None
    assert llm.completions.last_kwargs is None


@pytest.mark.asyncio
async def test_rag_image_with_caption_searches_caption_text(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    rag = FakeRAG()
    service = ChatService(repo, FakeLLM(), build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")
    part = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aW1n"}}

    _ = [
        chunk
        async for chunk in service.send_message(
            chat.id,
            "Что означает эта ошибка?",
            media_refs={"mime": "image/jpeg", "size": 3, "filename": "x.jpg", "part": part},
            rag_service=rag,
        )
    ]

    assert rag.question == "Что означает эта ошибка?"


@pytest.mark.asyncio
async def test_multi_part_media_all_parts_reach_provider_content(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    llm = FakeLLM()
    service = ChatService(repo, llm, build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")
    image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aW1n"}}
    image_two = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,bWlq"}}

    _ = [
        chunk
        async for chunk in service.send_message(
            chat.id,
            "скриншоты ошибки",
            media_refs={
                "mime": "image/jpeg",
                "size": 3,
                "filename": "a.jpg",
                "part": image,
                "parts": [image, image_two],
            },
        )
    ]

    user_message = next(message for message in llm.completions.last_kwargs["messages"] if message["role"] == "user")
    assert user_message["content"] == [
        {"type": "text", "text": "скриншоты ошибки"},
        image,
        image_two,
    ]


@pytest.mark.asyncio
async def test_multi_part_persisted_refs_keep_text_but_drop_image_payloads(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    service = ChatService(repo, FakeLLM(), build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")
    image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aW1n"}}
    doc_text = {"type": "text", "text": "[документ PDF]:\nсодержимое"}

    _ = [
        chunk
        async for chunk in service.send_message(
            chat.id,
            "материалы",
            media_refs={"mime": "image/jpeg", "part": image, "parts": [image, doc_text]},
        )
    ]

    stored = await repo.list_messages(chat.id)
    user_message = next(message for message in stored if message.role == "user")
    assert "image_url" not in str(user_message.media_refs)
    assert user_message.media_refs["parts"] == [doc_text]
    assert user_message.media_refs["part"] == doc_text


@pytest.mark.asyncio
async def test_rag_question_merges_all_extracted_text_parts(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    rag = FakeRAG()
    service = ChatService(repo, FakeLLM(), build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")
    doc_one = {"type": "text", "text": "[документ PDF]:\nпервая часть"}
    doc_two = {"type": "text", "text": "[документ DOCX]:\nвторая часть"}

    _ = [
        chunk
        async for chunk in service.send_message(
            chat.id,
            "сравни документы",
            media_refs={"mime": "application/pdf", "part": doc_one, "parts": [doc_one, doc_two]},
            rag_service=rag,
        )
    ]

    assert rag.question == "сравни документы\n[документ PDF]:\nпервая часть\n[документ DOCX]:\nвторая часть"


@pytest.mark.asyncio
async def test_rag_album_of_images_without_caption_still_refuses(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    llm = FakeLLM()
    rag = FakeRAG()
    service = ChatService(repo, llm, build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")
    image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aW1n"}}

    chunks = [
        chunk
        async for chunk in service.send_message(
            chat.id,
            "[фото]\n[фото]",
            media_refs={"mime": "image/jpeg", "part": image, "parts": [image, image]},
            rag_service=rag,
        )
    ]

    assert chunks == [MEDIA_IMAGE_NO_TEXT_REPLY]
    assert rag.question is None


@pytest.mark.asyncio
async def test_moderation_content_covers_all_text_parts(tmp_path: Path) -> None:
    doc_one = {"type": "text", "text": "текст первого документа"}
    doc_two = {"type": "text", "text": "текст второго документа"}

    merged = ChatService._moderation_content(
        "проверь",
        {"mime": "application/pdf", "part": doc_one, "parts": [doc_one, doc_two]},
    )

    assert merged == "проверь\nтекст первого документа\nтекст второго документа"
