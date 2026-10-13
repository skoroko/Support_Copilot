from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Update
from aiogram.types.update import UpdateTypeLookupError


class AdmissionMiddleware(BaseMiddleware):
    """Admit only named employees and approved conversations before any handler."""

    def __init__(self, *, user_ids: list[int], chat_ids: list[int]) -> None:
        self.user_ids = frozenset(user_ids)
        self.chat_ids = frozenset(chat_ids)

    async def __call__(
        self,
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        try:
            payload = event.event
        except UpdateTypeLookupError:
            return None
        user = getattr(payload, "from_user", None)
        if user is None or getattr(user, "is_bot", False) or user.id not in self.user_ids:
            return None

        chat = getattr(payload, "chat", None)
        if chat is None:
            chat = getattr(getattr(payload, "message", None), "chat", None)
        if chat is None:
            return None
        if chat.type == "private":
            if chat.id != user.id:
                return None
        elif chat.type not in {"group", "supergroup"} or chat.id not in self.chat_ids:
            return None
        return await handler(event, data)
