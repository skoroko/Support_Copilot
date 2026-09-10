from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from html import escape
import re
from uuid import UUID

import httpx
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.types import InlineKeyboardMarkup, Message

from bot.keyboards.inline import answer_kb
from bot.services.backend_client import BackendClient


BACKEND_EXCEPTIONS = (httpx.HTTPError,)
EDIT_DEBOUNCE_SECONDS = 0.8
RESPONSE_PLACEHOLDER = "Готовлю ответ…"
_INLINE_CODE_RE = re.compile(r"(`[^`\n]+`)")
_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


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
    if message_id is None:
        return None
    return answer_kb(chat_id, message_id, len(backend.last_response_sources(chat_id)))


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
        final_text += _format_cited_sources(final_text, sources_factory())
    final_html = _render_telegram_html(final_text)
    reply_markup = reply_markup_factory() if reply_markup_factory is not None else None
    if editable_response is not None:
        try:
            if reply_markup is None:
                await editable_response.edit_text(final_html, parse_mode=ParseMode.HTML)
            else:
                await editable_response.edit_text(
                    final_html,
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )
            return buffer
        except TelegramAPIError:
            pass
    if reply_markup is None:
        await message.bot.send_message(
            chat_id=message.chat.id,
            text=final_html,
            parse_mode=ParseMode.HTML,
        )
    else:
        await message.bot.send_message(
            chat_id=message.chat.id,
            text=final_html,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )
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


def _format_cited_sources(answer: str, sources: list[dict]) -> str:
    source_numbers = _cited_source_numbers(answer, len(sources))
    if not source_numbers:
        return ""
    source_items = [(number, sources[number - 1]) for number in source_numbers]
    return _format_source_items(source_items, heading="Источники")


def format_all_sources(sources: list[dict]) -> str:
    return _format_source_items(list(enumerate(sources, start=1)), heading="Все источники").removeprefix("\n\n")


def _cited_source_numbers(answer: str, source_count: int) -> list[int]:
    cited: set[int] = set()
    for match in _CITATION_RE.finditer(answer):
        cited.update(int(value.strip()) for value in match.group(1).split(","))
    return [number for number in range(1, source_count + 1) if number in cited]


def _format_source_items(source_items: list[tuple[int, dict]], *, heading: str) -> str:
    if not source_items:
        return ""
    lines = []
    for index, source in source_items:
        file_name = str(source.get("file_name") or "источник")
        page = source.get("page")
        page_text = f", стр. {page}" if page is not None else ""
        lines.append(f"[{index}] {file_name}{page_text}")
    return f"\n\n{heading}:\n" + "\n".join(lines)


def _render_telegram_html(text: str) -> str:
    """Render the small Markdown subset we request from the LLM as Telegram HTML.

    Telegram does not understand Markdown unless a parse mode is explicitly
    selected.  The streamed draft stays plain text because it is often
    incomplete; only the final answer is converted.  Escaping happens before
    inserting the two allowed tags, so generated text and document names
    cannot introduce arbitrary Telegram HTML.
    """

    rendered: list[str] = []
    for segment in _INLINE_CODE_RE.split(text):
        if segment.startswith("`") and segment.endswith("`"):
            rendered.append(f"<code>{escape(segment[1:-1])}</code>")
            continue
        escaped = escape(segment)
        rendered.append(_BOLD_RE.sub(r"<b>\1</b>", escaped))
    return "".join(rendered)
