from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID
from weakref import WeakValueDictionary

import pytest
from pydantic import SecretStr

from app.chat.deps import get_chat_service
from app.chat.domain import Chat, ChatMessage
from app.chat.repository import ChatRepository
from app.chat.service import (
    FALLBACK_SYSTEM_PROMPT,
    MODERATION_OUTPUT_REPLACEMENT,
    ChatNotFoundError,
    ChatService,
    MEDIA_IMAGE_NO_TEXT_REPLY,
    RAG_MEDIA_TEXT_MAX_CHARS,
    StreamReplace,
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
        self.system_prompt: str | None = None

    def answer(
        self,
        question: str,
        *,
        history: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        self.question = question
        self.history = history
        self.system_prompt = system_prompt
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


class FakeStreamingRAG:
    """RAG double exposing the ``stream_answer`` delta/final contract."""

    def __init__(
        self,
        deltas: list[str],
        final: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        self.deltas = deltas
        self.final = final
        self.sources = sources or []
        self.question: str | None = None
        self.system_prompt: str | None = None

    def stream_answer(
        self,
        question: str,
        *,
        history: list[dict[str, Any]],
        system_prompt: str | None = None,
    ):
        self.question = question
        self.system_prompt = system_prompt
        for delta in self.deltas:
            yield {"type": "delta", "delta": delta}
        yield {"type": "final", "answer": self.final, "sources": self.sources}


class BlockingStream:
    """Stream that emits one chunk, then blocks until ``release`` is set."""

    def __init__(self, content: str, release: asyncio.Event) -> None:
        self.content = content
        self.release = release
        self.sent = False

    def __aiter__(self) -> "BlockingStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self.sent:
            self.sent = True
            return {"choices": [{"delta": {"content": self.content}}]}
        await self.release.wait()
        raise StopAsyncIteration


class BlockingFirstLLM:
    """First ``create()`` returns a stream held on ``release``; later calls answer instantly."""

    def __init__(self, release: asyncio.Event) -> None:
        self.release = release
        self.started = asyncio.Event()
        self.calls: list[dict[str, Any]] = []

        async def create(**kwargs: Any) -> Any:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                self.started.set()
                return BlockingStream("ответ-1", self.release)
            return FakeStream(["ответ-2"])

        completions = type("Completions", (), {"create": staticmethod(create)})()
        self.chat = type("Chat", (), {"completions": completions})()


async def _collect_turn(
    service: ChatService,
    chat_id: UUID,
    content: str,
) -> list[str | StreamReplace]:
    return [chunk async for chunk in service.send_message(chat_id, content)]


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
async def test_non_rag_fallback_injects_default_system_prompt_when_chat_has_none(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    llm = FakeLLM()
    service = ChatService(repo, llm, build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")

    chunks = [chunk async for chunk in service.send_message(chat.id, "Привет")]

    assert chunks == ["Аня"]
    assert llm.completions.last_kwargs["messages"][0] == {
        "role": "system",
        "content": FALLBACK_SYSTEM_PROMPT,
    }


@pytest.mark.asyncio
async def test_non_rag_fallback_keeps_custom_system_prompt(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    llm = FakeLLM()
    service = ChatService(repo, llm, build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram", "custom support rules")

    _ = [chunk async for chunk in service.send_message(chat.id, "Привет")]

    messages = llm.completions.last_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "custom support rules"}
    assert all(message["content"] != FALLBACK_SYSTEM_PROMPT for message in messages)


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
async def test_rag_receives_chat_system_prompt_on_buffered_path(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    rag = FakeRAG()
    service = ChatService(repo, FakeLLM(), build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram", "Отвечай как дежурный инженер MIDAS.")

    chunks = [
        chunk
        async for chunk in service.send_message(chat.id, "Срок возврата?", rag_service=rag)
    ]

    assert chunks == ["Для электроники действует отдельный срок возврата [1]."]
    assert rag.system_prompt == "Отвечай как дежурный инженер MIDAS."


@pytest.mark.asyncio
async def test_rag_streaming_receives_chat_system_prompt(tmp_path: Path) -> None:
    repo = InMemoryRepository()
    rag = FakeStreamingRAG(deltas=["Ответ"], final="Ответ")
    service = ChatService(repo, FakeLLM(), build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram", "Держи ответы в двух абзацах.")

    _ = [
        chunk
        async for chunk in service.send_message(chat.id, "Вопрос", rag_service=rag)
    ]

    assert rag.system_prompt == "Держи ответы в двух абзацах."


@pytest.mark.asyncio
async def test_rag_without_chat_system_prompt_gets_none(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    rag = FakeRAG()
    service = ChatService(repo, FakeLLM(), build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")

    _ = [
        chunk
        async for chunk in service.send_message(chat.id, "Срок возврата?", rag_service=rag)
    ]

    assert rag.system_prompt is None


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
async def test_rag_streaming_yields_deltas_then_replaces_with_grounded(tmp_path: Path) -> None:
    repo = InMemoryRepository()
    sources = [{"id": "s1", "file_name": "returns.md", "page": 2, "score": 0.91, "snippet": "14 дней."}]
    rag = FakeStreamingRAG(
        deltas=["Черновик ", "ответа"],
        final="Для электроники действует отдельный срок возврата [1].",
        sources=sources,
    )
    service = ChatService(repo, FakeLLM(), build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")

    chunks = [
        chunk
        async for chunk in service.send_message(chat.id, "Срок возврата?", rag_service=rag)
    ]

    assert chunks[:2] == ["Черновик ", "ответа"]
    assert isinstance(chunks[-1], StreamReplace)
    assert chunks[-1].text == "Для электроники действует отдельный срок возврата [1]."
    assert rag.question == "Срок возврата?"
    stored = await repo.list_messages(chat.id)
    assert stored[-1].content == "Для электроники действует отдельный срок возврата [1]."
    assert stored[-1].rag_sources == sources
    assert service.last_response_sources(chat.id) == sources


@pytest.mark.asyncio
async def test_rag_streaming_skips_replace_when_draft_matches_final(tmp_path: Path) -> None:
    rag = FakeStreamingRAG(deltas=["Ответ ", "целиком"], final="Ответ целиком")
    service = ChatService(InMemoryRepository(), FakeLLM(), build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")

    chunks = [
        chunk
        async for chunk in service.send_message(chat.id, "Вопрос", rag_service=rag)
    ]

    assert chunks == ["Ответ ", "целиком"]
    assert not any(isinstance(chunk, StreamReplace) for chunk in chunks)


@pytest.mark.asyncio
async def test_rag_stream_blocked_fragment_replaces_shown_draft(tmp_path: Path) -> None:
    repo = InMemoryRepository()
    rag = FakeStreamingRAG(
        deltas=["Совет: ", "kill yourself", " — продолжение"],
        final="Совет: kill yourself — продолжение",
        sources=[{"id": "s1"}],
    )
    service = ChatService(repo, FakeLLM(), build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")

    chunks = [
        chunk
        async for chunk in service.send_message(chat.id, "обычный вопрос", rag_service=rag)
    ]

    assert chunks[0] == "Совет: "
    # The blocked delta and everything after it never reach the consumer.
    assert all("kill yourself" not in str(chunk) for chunk in chunks)
    assert isinstance(chunks[-1], StreamReplace)
    assert chunks[-1].text == MODERATION_OUTPUT_REPLACEMENT
    stored = await repo.list_messages(chat.id)
    assert stored[-1].content == MODERATION_OUTPUT_REPLACEMENT
    assert stored[-1].rag_sources == []
    assert service.last_response_sources(chat.id) == []


@pytest.mark.asyncio
async def test_moderation_content_covers_all_text_parts(tmp_path: Path) -> None:
    doc_one = {"type": "text", "text": "текст первого документа"}
    doc_two = {"type": "text", "text": "текст второго документа"}

    merged = ChatService._moderation_content(
        "проверь",
        {"mime": "application/pdf", "part": doc_one, "parts": [doc_one, doc_two]},
    )

    assert merged == "проверь\nтекст первого документа\nтекст второго документа"


@pytest.mark.asyncio
async def test_concurrent_turns_serialize_and_keep_pair_order(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    release = asyncio.Event()
    llm = BlockingFirstLLM(release)
    service = ChatService(repo, llm, build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")

    first = asyncio.ensure_future(_collect_turn(service, chat.id, "первый вопрос"))
    await llm.started.wait()
    second = asyncio.ensure_future(_collect_turn(service, chat.id, "второй вопрос"))
    for _ in range(5):
        await asyncio.sleep(0)

    # The second turn is queued behind the in-flight one: nothing appended yet.
    assert await repo.list_messages(chat.id) == []

    release.set()
    first_chunks, second_chunks = await asyncio.gather(first, second)

    assert first_chunks == ["ответ-1"]
    assert second_chunks == ["ответ-2"]
    stored = await repo.list_messages(chat.id)
    assert [(message.role, message.content) for message in stored] == [
        ("user", "первый вопрос"),
        ("assistant", "ответ-1"),
        ("user", "второй вопрос"),
        ("assistant", "ответ-2"),
    ]
    # The serialized second turn read history that already contained the
    # first pair instead of the same pre-turn snapshot as the first send.
    second_contents = [message["content"] for message in llm.calls[1]["messages"]]
    assert "первый вопрос" in second_contents
    assert "ответ-1" in second_contents
    assert "второй вопрос" in second_contents


@pytest.mark.asyncio
async def test_concurrent_turns_in_different_chats_do_not_block(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    release = asyncio.Event()
    llm = BlockingFirstLLM(release)
    service = ChatService(repo, llm, build_settings(tmp_path))
    chat_one = await service.create_chat("telegram-1", "telegram")
    chat_two = await service.create_chat("telegram-2", "telegram")

    first = asyncio.ensure_future(_collect_turn(service, chat_one.id, "вопрос A"))
    await llm.started.wait()

    second_chunks = await asyncio.wait_for(
        _collect_turn(service, chat_two.id, "вопрос B"),
        timeout=5,
    )

    assert second_chunks == ["ответ-2"]
    assert await repo.list_messages(chat_one.id) == []
    release.set()
    assert await first == ["ответ-1"]


@pytest.mark.asyncio
async def test_clear_history_waits_for_in_flight_turn(tmp_path: Path) -> None:
    repo: ChatRepository = InMemoryRepository()
    release = asyncio.Event()
    llm = BlockingFirstLLM(release)
    service = ChatService(repo, llm, build_settings(tmp_path))
    chat = await service.create_chat("telegram-1", "telegram")

    first = asyncio.ensure_future(_collect_turn(service, chat.id, "вопрос"))
    await llm.started.wait()
    cleared = asyncio.ensure_future(service.clear_history(chat.id))
    for _ in range(5):
        await asyncio.sleep(0)
    assert not cleared.done()

    release.set()
    await asyncio.gather(first, cleared)

    # The turn persisted first; clear ran after it, so history is empty again.
    assert await repo.list_messages(chat.id) == []


@pytest.mark.asyncio
async def test_shared_turn_locks_serialize_across_service_instances(tmp_path: Path) -> None:
    # Mirrors production wiring: per-request ChatService instances share the
    # turn-lock registry, so concurrent sends still serialize per chat.
    repo: ChatRepository = InMemoryRepository()
    release = asyncio.Event()
    llm = BlockingFirstLLM(release)
    settings = build_settings(tmp_path)
    turn_locks: WeakValueDictionary[UUID, asyncio.Lock] = WeakValueDictionary()
    service_one = ChatService(repo, llm, settings, turn_locks=turn_locks)
    service_two = ChatService(repo, llm, settings, turn_locks=turn_locks)
    chat = await service_one.create_chat("telegram-1", "telegram")

    first = asyncio.ensure_future(_collect_turn(service_one, chat.id, "первый"))
    await llm.started.wait()
    second = asyncio.ensure_future(_collect_turn(service_two, chat.id, "второй"))
    for _ in range(5):
        await asyncio.sleep(0)
    assert await repo.list_messages(chat.id) == []

    release.set()
    await asyncio.gather(first, second)
    stored = await repo.list_messages(chat.id)
    assert [message.content for message in stored] == ["первый", "ответ-1", "второй", "ответ-2"]


def test_get_chat_service_shares_turn_locks_via_app_state(tmp_path: Path) -> None:
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    settings = build_settings(tmp_path)

    service_one = get_chat_service(request, InMemoryRepository(), FakeLLM(), settings)
    service_two = get_chat_service(request, InMemoryRepository(), FakeLLM(), settings)

    assert service_one._turn_locks is service_two._turn_locks
