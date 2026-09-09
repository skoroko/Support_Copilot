from __future__ import annotations

from aiogram import F, Router
from aiogram.types import Message

from bot.handlers._common import (
    BACKEND_EXCEPTIONS,
    backend_error_message,
    get_telegram_chat_id,
    response_feedback_markup,
    stream_to_chat,
)
from bot.services.backend_client import BackendClient


router = Router()


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: Message, backend: BackendClient) -> None:
    try:
        chat_id = await get_telegram_chat_id(message, backend)
        await stream_to_chat(
            message,
            backend.send_message(chat_id, message.text or ""),
            reply_markup_factory=lambda: response_feedback_markup(backend, chat_id),
        )
    except BACKEND_EXCEPTIONS as exc:
        await message.answer(backend_error_message(exc))
