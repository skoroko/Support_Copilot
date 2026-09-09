from __future__ import annotations

from uuid import UUID

from aiogram import F, Router
from aiogram.types import CallbackQuery

from bot.handlers._common import BACKEND_EXCEPTIONS, get_telegram_chat_id
from bot.services.backend_client import BackendClient


router = Router()


@router.callback_query(F.data.startswith("fb:"))
async def save_feedback(callback: CallbackQuery, backend: BackendClient) -> None:
    if callback.message is None or callback.from_user is None:
        await callback.answer("Не удалось сохранить оценку.", show_alert=True)
        return
    try:
        _, vote, raw_message_id = (callback.data or "").split(":", maxsplit=2)
        if vote not in {"up", "down"}:
            raise ValueError("invalid vote")
        message_id = UUID(raw_message_id)
        chat_id = await get_telegram_chat_id(callback.message, backend)
        await backend.save_feedback(chat_id, message_id, vote)
    except (ValueError, *BACKEND_EXCEPTIONS):
        await callback.answer("Не удалось сохранить оценку. Попробуйте позже.", show_alert=True)
        return

    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer("Спасибо за оценку!")
