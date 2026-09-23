from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import pytest

from bot.handlers._common import backend_error_message
from bot.handlers.admin import IsAdmin
from bot.handlers.feedback import save_feedback
from bot.keyboards.inline import (
    answer_kb,
    decode_message_context,
    encode_message_context,
    feedback_kb,
)
from bot.services.backend_client import BackendClient


@pytest.mark.asyncio
async def test_is_admin_uses_router_level_id_set() -> None:
    admin = SimpleNamespace(from_user=SimpleNamespace(id=42))
    stranger = SimpleNamespace(from_user=SimpleNamespace(id=7))
    check = IsAdmin([42])

    assert await check(admin) is True
    assert await check(stranger) is False


def test_feedback_keyboard_uses_required_callback_contract() -> None:
    message_id = UUID("00000000-0000-0000-0000-000000000123")
    markup = feedback_kb(message_id)
    callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
    assert callbacks == [f"fb:up:{message_id}", f"fb:down:{message_id}"]


def test_answer_keyboard_compacts_chat_and_message_ids_below_telegram_limit() -> None:
    chat_id = UUID("00000000-0000-0000-0000-000000000111")
    message_id = UUID("00000000-0000-0000-0000-000000000123")
    markup = answer_kb(chat_id, message_id, source_count=10)
    callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
    texts = [button.text for row in markup.inline_keyboard for button in row]

    assert len(callbacks) == 3
    assert texts[0] == "📚 Документы (10)"
    assert all(callback is not None and len(callback.encode("utf-8")) <= 64 for callback in callbacks)
    assert callbacks[0].startswith("src:")
    assert decode_message_context(callbacks[0].removeprefix("src:")) == (chat_id, message_id)
    assert callbacks[1].startswith("fb:u:")
    assert callbacks[2].startswith("fb:d:")


def _status_error(status: int, payload: dict) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://backend/chats/x/messages")
    response = httpx.Response(status, request=request, json=payload)
    return httpx.HTTPStatusError("error", request=request, response=response)


def test_moderation_403_has_user_friendly_message() -> None:
    exc = _status_error(403, {"error": {"code": "moderation_blocked", "categories": ["violence"]}})
    message = backend_error_message(exc)
    assert "правил" in message


def test_overlong_input_400_maps_to_length_hint() -> None:
    exc = _status_error(
        400,
        {"error": {"code": "security_input_blocked", "rule": "length", "limit": 16000}},
    )
    message = backend_error_message(exc)
    assert "длинн" in message
    assert "16000" in message


def test_injection_400_maps_to_instruction_hint() -> None:
    exc = _status_error(400, {"error": {"code": "security_input_blocked", "rule": "injection"}})
    message = backend_error_message(exc)
    assert "инструкц" in message

def test_encoded_injection_400_maps_to_decoded_hint() -> None:
    exc = _status_error(400, {"error": {"code": "security_input_blocked", "rule": "encoded_injection"}})
    message = backend_error_message(exc)
    assert "закодированн" in message

def test_encoding_400_maps_to_charset_hint() -> None:
    exc = _status_error(400, {"error": {"code": "security_input_blocked", "rule": "encoding"}})
    message = backend_error_message(exc)
    assert "нечитаем" in message


def test_media_too_large_413_maps_to_size_hint() -> None:
    exc = _status_error(413, {"error": {"code": "media_too_large", "message": "Файл превышает лимит 10 МиБ."}})
    assert "10 МБ" in backend_error_message(exc)


def test_unsupported_media_415_maps_to_format_hint() -> None:
    exc = _status_error(
        415,
        {"error": {"code": "unsupported_media_type", "message": "Unsupported media type: text/plain"}},
    )
    message = backend_error_message(exc)
    assert "формат" in message.lower()
    assert "PDF/DOCX" in message


def test_media_processing_422_surfaces_server_reason() -> None:
    exc = _status_error(
        422,
        {"error": {"code": "media_processing_failed", "message": "PDF похож на скан: для него требуется OCR."}},
    )
    assert backend_error_message(exc) == "PDF похож на скан: для него требуется OCR."


def test_media_processing_422_without_reason_uses_fallback() -> None:
    exc = _status_error(422, {"error": {"code": "media_processing_failed"}})
    assert "прочитать" in backend_error_message(exc)


class _FeedbackBackend:
    def __init__(self) -> None:
        self.calls: dict[str, str | None] = {}

    async def get_message_sources(self, chat_id, message_id, *, owner_external_id=None):
        self.calls["sources_owner"] = owner_external_id
        return [{"file_name": "manual.pdf"}]

    async def save_feedback(self, chat_id, message_id, value, shown_sources=None, owner_external_id=None):
        self.calls["feedback_owner"] = owner_external_id
        return {}


def _feedback_callback(*, chat_id: int, chat_type: str | None, user_id: int) -> SimpleNamespace:
    chat = SimpleNamespace(id=chat_id)
    if chat_type is not None:
        chat.type = chat_type
    return SimpleNamespace(
        data=f"fb:u:{encode_message_context(UUID(int=1), UUID(int=2))}",
        message=SimpleNamespace(chat=chat, edit_reply_markup=AsyncMock()),
        from_user=SimpleNamespace(id=user_id),
        answer=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_group_feedback_uses_clicker_scoped_owner() -> None:
    backend = _FeedbackBackend()
    callback = _feedback_callback(chat_id=-100, chat_type="supergroup", user_id=7)

    await save_feedback(callback, backend)  # type: ignore[arg-type]

    assert backend.calls["sources_owner"] == "-100:7"
    assert backend.calls["feedback_owner"] == "-100:7"


@pytest.mark.asyncio
async def test_private_feedback_uses_chat_id_owner() -> None:
    backend = _FeedbackBackend()
    callback = _feedback_callback(chat_id=42, chat_type="private", user_id=42)

    await save_feedback(callback, backend)  # type: ignore[arg-type]

    assert backend.calls["sources_owner"] == "42"
    assert backend.calls["feedback_owner"] == "42"


@pytest.mark.asyncio
async def test_admin_client_sends_admin_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Admin-Token"] == "secret"
        return httpx.Response(200, json={"total_messages": 0})

    async with httpx.AsyncClient(base_url="http://backend", transport=httpx.MockTransport(handler)) as http:
        client = BackendClient(
            "http://backend",
            client=http,
            internal_token="test-internal-token-0000000000000001",
            admin_token="secret",
        )
        assert await client.get_admin_stats() == {"total_messages": 0}
