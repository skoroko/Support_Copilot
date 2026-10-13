"""Small native navigation panels; callback IDs contain no backend identities."""

import re

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

_CALLBACK = re.compile(r"mn:1:([1-9][0-9]{0,19}):(home|examples|help|history|pending|new|reset)(?::([A-Za-z0-9_-]{12}))?\Z")


def menu_data(user_id: int, action: str, token: str | None = None) -> str:
    return f"mn:1:{user_id}:{action}" + (f":{token}" if token else "")


def parse_menu_data(data: str) -> tuple[int, str, str | None]:
    match = _CALLBACK.fullmatch(data)
    if not match or (match[2] == "reset") != (match[3] is not None):
        raise ValueError("invalid menu callback")
    return int(match[1]), match[2], match[3]


def home_kb(user_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for text, action in (
        ("💡 Примеры вопросов", "examples"), ("📖 Как пользоваться", "help"),
        ("🕘 История", "history"), ("📝 Ожидающие отправки", "pending"),
        ("🆕 Новый диалог", "new"),
    ):
        builder.button(text=text, callback_data=menu_data(user_id, action))
    builder.adjust(2, 2, 1)
    return builder.as_markup()


def back_kb(user_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="← В меню", callback_data=menu_data(user_id, "home"))
    return builder.as_markup()


def reset_kb(user_id: int, token: str) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="Да, начать заново", callback_data=menu_data(user_id, "reset", token))
    builder.button(text="Отмена", callback_data=menu_data(user_id, "home"))
    builder.adjust(1)
    return builder.as_markup()
