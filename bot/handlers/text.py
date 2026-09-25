from __future__ import annotations

from aiogram import F, Router
from aiogram.types import Message

from bot.handlers import _burst
from bot.handlers._common import compose_message_text, is_bot_or_channel_edit
from bot.services.backend_client import BackendClient

router = Router()


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: Message, backend: BackendClient) -> None:
    _burst.enqueue_text(message, backend)


@router.edited_message(F.text & ~F.text.startswith("/"))
async def handle_edited_text(message: Message, backend: BackendClient) -> None:
    if is_bot_or_channel_edit(message):
        return
    item = _burst.BurstItem(message=message, text=compose_message_text(message))
    if _burst.absorb_edit(message, backend, item):
        return
    await _burst.deliver_now(item, backend)
