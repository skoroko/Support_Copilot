from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

import httpx
from aiogram.types import Message

from bot.services.backend_client import BackendClient


BACKEND_ERROR_MESSAGE = (
    "Backend Support Copilot сейчас недоступен. "
    "Проверьте, что FastAPI-сервис запущен, и попробуйте еще раз."
)

BACKEND_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.TimeoutException,
    httpx.HTTPStatusError,
    httpx.HTTPError,
)


async def get_telegram_chat_id(message: Message, backend: BackendClient) -> UUID:
    return await backend.get_or_create_chat(str(message.chat.id), "telegram")


async def render_stream(message: Message, chunks: AsyncIterator[str]) -> None:
    answer = await message.answer("Готовлю ответ...")
    buffer = ""
    async for chunk in chunks:
        buffer += chunk
        await answer.edit_text(buffer)
    if not buffer:
        await answer.edit_text("Backend вернул пустой ответ.")
