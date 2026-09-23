from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from aiogram.types import Chat as TgChat
from aiogram.types import Message as TgMessage
from aiogram.types import MessageEntity

from bot.handlers import _burst
from bot.handlers._common import compose_message_text, message_text, quote_block
from bot.handlers.media import handle_edited_media, handle_media
from bot.handlers.text import handle_edited_text, handle_text

CHAT_ID = 42
BACKEND_CHAT_ID = UUID("00000000-0000-0000-0000-000000000777")


class FakeBackend:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def get_or_create_chat(self, owner_external_id: str, interface: str) -> UUID:
        return BACKEND_CHAT_ID

    def send_message(self, chat_id, content, media=None):
        self.sent.append({"content": content, "media": media})

        async def gen():
            yield "Ответ"

        return gen()

    def last_response_message_id(self, chat_id):
        return None

    def last_response_sources(self, chat_id):
        return []


def make_bot(download: bytes = b"file-bytes") -> SimpleNamespace:
    async def _download_file(file_path, destination):
        destination.write(download)

    return SimpleNamespace(
        send_chat_action=AsyncMock(),
        send_message_draft=AsyncMock(),
        send_message=AsyncMock(),
        get_file=AsyncMock(return_value=SimpleNamespace(file_path="files/x")),
        download_file=AsyncMock(side_effect=_download_file),
    )


def make_message(
    *,
    message_id: int = 1,
    text: str | None = None,
    caption: str | None = None,
    reply_to: SimpleNamespace | None = None,
    media_group_id: str | None = None,
    photo: list | None = None,
    document=None,
    voice=None,
    audio=None,
    is_bot: bool = False,
    bot=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(id=CHAT_ID),
        message_id=message_id,
        text=text,
        caption=caption,
        entities=None,
        caption_entities=None,
        reply_to_message=reply_to,
        media_group_id=media_group_id,
        from_user=SimpleNamespace(is_bot=is_bot),
        photo=photo,
        voice=voice,
        audio=audio,
        document=document,
        video=None,
        video_note=None,
        sticker=None,
        location=None,
        venue=None,
        contact=None,
        poll=None,
        bot=bot or make_bot(),
        answer=AsyncMock(return_value=SimpleNamespace(edit_text=AsyncMock())),
    )


def make_state(state: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(get_state=AsyncMock(return_value=state))


@pytest.fixture(autouse=True)
def fast_debounce(monkeypatch):
    monkeypatch.setattr(_burst, "TEXT_DEBOUNCE_SECONDS", 0.05)
    monkeypatch.setattr(_burst, "MEDIA_GROUP_DEBOUNCE_SECONDS", 0.1)
    monkeypatch.setattr(_burst, "MAX_BATCH_AGE_SECONDS", 30.0)
    yield
    for batch in _burst._pending.values():
        if batch.task is not None:
            batch.task.cancel()
    _burst._pending.clear()


async def wait_flush(extra: float = 0.3) -> None:
    import asyncio

    await asyncio.sleep(extra)


@pytest.mark.asyncio
async def test_quick_text_messages_merge_into_one_request() -> None:
    backend = FakeBackend()
    await handle_text(make_message(message_id=1, text="Как включить"), backend)
    await handle_text(make_message(message_id=2, text="режим совместимости?"), backend)
    await wait_flush()

    assert len(backend.sent) == 1
    assert backend.sent[0]["content"] == "Как включить\nрежим совместимости?"
    assert backend.sent[0]["media"] is None


@pytest.mark.asyncio
async def test_slow_messages_stay_separate_requests() -> None:
    backend = FakeBackend()
    await handle_text(make_message(message_id=1, text="первый"), backend)
    await wait_flush()
    await handle_text(make_message(message_id=2, text="второй"), backend)
    await wait_flush()

    assert [sent["content"] for sent in backend.sent] == ["первый", "второй"]


def test_text_link_entity_is_inlined_as_url() -> None:
    message = TgMessage(
        message_id=1,
        date=datetime.now(UTC),
        chat=TgChat(id=1, type="private"),
        text="смотри документ и инструкцию",
        entities=[
            MessageEntity(type="text_link", offset=7, length=8, url="https://docs.example.com/a"),
            MessageEntity(type="bold", offset=19, length=10),
        ],
    )

    assert message_text(message) == (
        "смотри [документ](https://docs.example.com/a) и инструкцию"
    )


def test_text_link_offsets_are_utf16_safe_with_emoji() -> None:
    message = TgMessage(
        message_id=1,
        date=datetime.now(UTC),
        chat=TgChat(id=1, type="private"),
        text="😀 лог",
        entities=[MessageEntity(type="text_link", offset=3, length=3, url="https://x.io")],
    )

    assert message_text(message) == "😀 [лог](https://x.io)"


def test_quote_block_uses_replied_text() -> None:
    reply = make_message(message_id=10, text="клиент спрашивает про лицензию")
    message = make_message(message_id=11, text="что ответить?", reply_to=reply)

    content = compose_message_text(message)

    assert "[сообщение, на которое отвечает пользователь]:" in content
    assert "клиент спрашивает про лицензию" in content
    assert content.endswith("что ответить?")


def test_quote_block_describes_replied_attachment() -> None:
    reply = make_message(message_id=10, photo=[SimpleNamespace(file_id="p", file_size=10)])
    message = make_message(message_id=11, text="а это что?", reply_to=reply)

    assert "[фото]" in quote_block(message)


@pytest.mark.asyncio
async def test_reply_quote_reaches_backend_content() -> None:
    backend = FakeBackend()
    reply = make_message(message_id=10, text="обращение клиента")
    await handle_text(make_message(message_id=11, text="помоги ответить", reply_to=reply), backend)
    await wait_flush()

    assert len(backend.sent) == 1
    assert "обращение клиента" in backend.sent[0]["content"]


@pytest.mark.asyncio
async def test_edit_of_pending_message_replaces_its_text() -> None:
    backend = FakeBackend()
    state = make_state()
    await handle_text(make_message(message_id=5, text="как настроить filtr"), backend)
    await handle_edited_text(make_message(message_id=5, text="как настроить фильтр"), backend, state)
    await wait_flush()

    assert len(backend.sent) == 1
    assert backend.sent[0]["content"] == "как настроить фильтр"


@pytest.mark.asyncio
async def test_edit_of_flushed_message_sends_new_request() -> None:
    backend = FakeBackend()
    state = make_state()
    await handle_text(make_message(message_id=5, text="исходный вопрос"), backend)
    await wait_flush()
    await handle_edited_text(make_message(message_id=5, text="исправленный вопрос"), backend, state)
    await wait_flush()

    assert [sent["content"] for sent in backend.sent] == ["исходный вопрос", "исправленный вопрос"]


@pytest.mark.asyncio
async def test_bot_own_edits_are_ignored() -> None:
    backend = FakeBackend()
    state = make_state()
    await handle_edited_text(
        make_message(message_id=9, text="Готовлю ответ…", is_bot=True), backend, state
    )
    await wait_flush()

    assert backend.sent == []


@pytest.mark.asyncio
async def test_edit_is_ignored_during_fsm_question_flow() -> None:
    backend = FakeBackend()
    state = make_state("AskFlow:waiting_for_question")
    await handle_edited_text(make_message(message_id=9, text="edit"), backend, state)
    await wait_flush()

    assert backend.sent == []


@pytest.mark.asyncio
async def test_media_group_merges_into_one_request_with_all_files() -> None:
    backend = FakeBackend()
    group = "album-1"
    first = make_message(
        message_id=1,
        media_group_id=group,
        photo=[SimpleNamespace(file_id="p1", file_size=100)],
        caption="скриншоты ошибки",
    )
    second = make_message(
        message_id=2,
        media_group_id=group,
        photo=[SimpleNamespace(file_id="p2", file_size=200)],
    )
    await handle_media(first, backend)
    await handle_media(second, backend)
    await wait_flush()

    assert len(backend.sent) == 1
    sent = backend.sent[0]
    assert sent["media"] == [(b"file-bytes", "image/jpeg"), (b"file-bytes", "image/jpeg")]
    assert "скриншоты ошибки" in sent["content"]
    assert "[фото]" in sent["content"]


@pytest.mark.asyncio
async def test_text_then_photo_merge_into_single_request() -> None:
    backend = FakeBackend()
    await handle_text(make_message(message_id=1, text="вот ошибка"), backend)
    await handle_media(
        make_message(message_id=2, photo=[SimpleNamespace(file_id="p1", file_size=100)]),
        backend,
    )
    await wait_flush()

    assert len(backend.sent) == 1
    assert backend.sent[0]["content"] == "вот ошибка\n[фото]"
    assert backend.sent[0]["media"] == [(b"file-bytes", "image/jpeg")]


@pytest.mark.asyncio
async def test_invalid_media_reports_error_once_without_request() -> None:
    backend = FakeBackend()
    document = SimpleNamespace(file_id="d1", file_name="virus.exe", file_size=10, mime_type=None)
    await handle_media(make_message(message_id=1, document=document), backend)
    await wait_flush()

    assert backend.sent == []


@pytest.mark.asyncio
async def test_caption_edit_on_pending_media_updates_content() -> None:
    backend = FakeBackend()
    photo = [SimpleNamespace(file_id="p1", file_size=100)]
    await handle_media(make_message(message_id=7, photo=photo), backend)
    edited = make_message(message_id=7, photo=photo, caption="ошибка при сборке")
    await handle_edited_media(edited, backend)
    await wait_flush()

    assert len(backend.sent) == 1
    assert backend.sent[0]["content"] == "ошибка при сборке"


@pytest.mark.asyncio
async def test_flush_pending_delivers_immediately() -> None:
    backend = FakeBackend()
    await handle_text(make_message(message_id=1, text="срочный вопрос"), backend)

    await _burst.flush_pending(CHAT_ID, backend)

    assert len(backend.sent) == 1
    assert backend.sent[0]["content"] == "срочный вопрос"
    assert CHAT_ID not in _burst._pending
