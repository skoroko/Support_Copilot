from __future__ import annotations

import base64
import binascii
from uuid import UUID

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def feedback_kb(message_id: UUID | str) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="👍", callback_data=f"fb:up:{message_id}")
    builder.button(text="👎", callback_data=f"fb:down:{message_id}")
    builder.adjust(2)
    return builder.as_markup()


def answer_kb(chat_id: UUID | str, message_id: UUID | str, source_count: int) -> InlineKeyboardMarkup:
    """Keyboard for a generated answer without placing document data in callbacks."""

    token = encode_message_context(chat_id, message_id)
    builder = InlineKeyboardBuilder()
    if source_count > 0:
        builder.button(text=f"📚 Документы ({source_count})", callback_data=f"src:{token}")
    builder.button(text="👍", callback_data=f"fb:u:{token}")
    builder.button(text="👎", callback_data=f"fb:d:{token}")
    builder.adjust(1, 2) if source_count > 0 else builder.adjust(2)
    return builder.as_markup()


def approval_kb(chat_id: UUID | str, approval_id: UUID | str) -> InlineKeyboardMarkup:
    context = encode_message_context(chat_id, approval_id)
    builder = InlineKeyboardBuilder()
    builder.button(text="Подтвердить", callback_data=f"ap:y:{context}")
    builder.button(text="Отменить", callback_data=f"ap:n:{context}")
    builder.adjust(2)
    return builder.as_markup()


def sources_kb(chat_id: UUID | str, message_id: UUID | str, source_count: int) -> InlineKeyboardMarkup | None:
    if source_count < 1:
        return None
    builder = InlineKeyboardBuilder()
    builder.button(
        text=f"📚 Документы ({source_count})",
        callback_data=f"src:{encode_message_context(chat_id, message_id)}",
    )
    return builder.as_markup()


def encode_message_context(chat_id: UUID | str, message_id: UUID | str) -> str:
    raw = UUID(str(chat_id)).bytes + UUID(str(message_id)).bytes
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_message_context(token: str) -> tuple[UUID, UUID]:
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
        raise ValueError("invalid message context") from exc
    if len(raw) != 32:
        raise ValueError("invalid message context")
    return UUID(bytes=raw[:16]), UUID(bytes=raw[16:])
