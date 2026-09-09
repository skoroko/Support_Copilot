from __future__ import annotations

from uuid import UUID

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


SUPPORT_TOPICS: dict[str, str] = {
    "billing": "Биллинг",
    "installation": "Установка",
    "api_errors": "Ошибки API",
    "license": "Лицензирование",
    "escalation": "Эскалация",
}


def topics_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for slug, title in SUPPORT_TOPICS.items():
        builder.button(text=title, callback_data=f"topic:{slug}")
    builder.button(text="Отмена", callback_data="topic:cancel")
    builder.adjust(1)
    return builder.as_markup()


def feedback_kb(message_id: UUID | str) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="👍", callback_data=f"fb:up:{message_id}")
    builder.button(text="👎", callback_data=f"fb:down:{message_id}")
    builder.adjust(2)
    return builder.as_markup()
