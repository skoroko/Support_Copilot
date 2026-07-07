from __future__ import annotations

from aiogram import F, Router
from aiogram.types import Message

from bot.handlers._common import BACKEND_ERROR_MESSAGE, BACKEND_EXCEPTIONS, get_telegram_chat_id, render_stream
from bot.services.backend_client import BackendClient


router = Router()


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: Message, backend: BackendClient) -> None:
    try:
        chat_id = await get_telegram_chat_id(message, backend)
        await render_stream(message, backend.send_message(chat_id, message.text or ""))
    except BACKEND_EXCEPTIONS:
        await message.answer(BACKEND_ERROR_MESSAGE)
