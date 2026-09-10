from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from uuid import UUID

import httpx
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.types import InlineKeyboardMarkup, Message

from bot.keyboards.inline import feedback_kb
from bot.services.backend_client import BackendClient


BACKEND_EXCEPTIONS = (httpx.HTTPError,)
EDIT_DEBOUNCE_SECONDS = 0.8
RESPONSE_PLACEHOLDER = "Готовлю ответ…"


async def get_telegram_chat_id(message: Message, backend: BackendClient) -> UUID:
    return await backend.get_or_create_chat(str(message.chat.id), "telegram")


def backend_error_message(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.ConnectError):
        return "Сервис недоступен, попробуйте позже."
    if isinstance(exc, httpx.ReadTimeout):
        return "Ответ занимает слишком долго. Попробуйте ещё раз позже."
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 403:
            try:
                payload = exc.response.json()
                error = payload.get("error", payload.get("detail", {}))
            except ValueError:
                error = {}
            if error.get("code") == "moderation_blocked":
                return "Не могу обработать этот запрос — он мог нарушить правила."
        if exc.response.status_code == 429:
            return "Слишком много запросов, подождите минуту."
        if exc.response.status_code >= 500:
            return "Внутренняя ошибка сервиса. Попробуйте позже."
    return "Не удалось обработать запрос. Проверьте данные и попробуйте ещё раз."


async def _typing(message: Message, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
        except TelegramAPIError:
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            continue


def response_feedback_markup(backend: BackendClient, chat_id: UUID) -> InlineKeyboardMarkup | None:
    message_id = backend.last_response_message_id(chat_id)
    return feedback_kb(message_id) if message_id is not None else None


async def stream_to_chat(
    message: Message,
    tokens: AsyncIterator[str],
    *,
    reply_markup_factory: Callable[[], InlineKeyboardMarkup | None] | None = None,
    sources_factory: Callable[[], list[dict]] | None = None,
) -> str:
    draft_id = uuid.uuid4().int & 0xFFFFFFFF
    buffer = ""
    last_edit_at = 0.0
    draft_paused_until = 0.0
    editable_response = await _start_editable_response(message)
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_typing(message, stop_typing))
    try:
        async for delta in tokens:
            if not delta:
                continue
            if not buffer:
                stop_typing.set()
            buffer += delta
            now = time.monotonic()
            if not buffer.strip() or now < draft_paused_until or now - last_edit_at < EDIT_DEBOUNCE_SECONDS:
                continue
            if editable_response is not None:
                try:
                    await editable_response.edit_text(buffer)
                    last_edit_at = now
                    continue
                except TelegramRetryAfter as exc:
                    draft_paused_until = time.monotonic() + exc.retry_after
                    continue
                except TelegramAPIError:
                    editable_response = None
            try:
                await message.bot.send_message_draft(chat_id=message.chat.id, text=buffer, draft_id=draft_id)
                last_edit_at = now
            except TelegramRetryAfter as exc:
                draft_paused_until = time.monotonic() + exc.retry_after
    finally:
        stop_typing.set()
        await typing_task

    final_text = buffer or "Backend вернул пустой ответ."
    if sources_factory is not None:
        final_text += _format_sources(sources_factory())
    reply_markup = reply_markup_factory() if reply_markup_factory is not None else None
    if editable_response is not None:
        try:
            if reply_markup is None:
                await editable_response.edit_text(final_text)
            else:
                await editable_response.edit_text(final_text, reply_markup=reply_markup)
            return buffer
        except TelegramAPIError:
            pass
    if reply_markup is None:
        await message.bot.send_message(chat_id=message.chat.id, text=final_text)
    else:
        await message.bot.send_message(chat_id=message.chat.id, text=final_text, reply_markup=reply_markup)
    return buffer


render_stream = stream_to_chat


async def _start_editable_response(message: Message) -> Message | None:
    """Create one bot response to update in-place while SSE tokens arrive.

    The fallback intentionally keeps the old draft/final-message path for tests
    and for transports that cannot return an editable Telegram message.
    """

    answer = getattr(message, "answer", None)
    if not callable(answer):
        return None
    try:
        return await answer(RESPONSE_PLACEHOLDER)
    except TelegramAPIError:
        return None


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = []
    for index, source in enumerate(sources[:5], start=1):
        file_name = str(source.get("file_name") or "источник")
        page = source.get("page")
        page_text = f", стр. {page}" if page is not None else ""
        lines.append(f"[{index}] {file_name}{page_text}")
    return "\n\nИсточники:\n" + "\n".join(lines)
