from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from html import escape
import re
from typing import Any
from uuid import UUID

import httpx
from aiogram.enums import ChatAction, ChatType, ParseMode
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message
from aiogram.utils.text_decorations import add_surrogates, remove_surrogates

from bot.keyboards.inline import answer_kb
from bot.services.backend_client import BackendClient, StreamReplace


BACKEND_EXCEPTIONS = (httpx.HTTPError,)
EDIT_DEBOUNCE_SECONDS = 0.8
QUOTE_MAX_CHARS = 1500
RESPONSE_PLACEHOLDER = "Готовлю ответ…"
_INLINE_CODE_RE = re.compile(r"(`[^`\n]+`)")
_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_MEDIA_ERROR_MESSAGES = {
    "media_too_large": "Файл слишком большой. Максимальный размер — 10 МБ.",
    "unsupported_media_type": (
        "Этот формат файла не поддерживается. "
        "Отправьте фото (JPEG, PNG, WebP), документ PDF/DOCX или голосовое сообщение."
    ),
    "media_processing_failed": "Не удалось прочитать файл. Проверьте, что он не повреждён, и попробуйте ещё раз.",
}
_SECURITY_RULE_MESSAGES = {
    "injection": (
        "Сообщение похоже на попытку изменить инструкции ассистента — "
        "оно не может быть обработано."
    ),
    "encoded_injection": (
        "В сообщении есть закодированный фрагмент со служебными инструкциями. "
        "Уберите его или перешлите содержимое открытым текстом."
    ),
    "encoding": (
        "Сообщение содержит слишком много нечитаемых символов — "
        "проверьте текст и попробуйте ещё раз."
    ),
}
_SECURITY_RULE_DEFAULT = (
    "Сообщение отклонено проверкой безопасности. "
    "Уберите из него служебные инструкции и закодированные фрагменты."
)


def _is_private_chat_type(chat_type: object) -> bool:
    return chat_type is None or chat_type == ChatType.PRIVATE


def is_group_chat(message: Message) -> bool:
    """Non-private chats get per-sender history and reply attribution."""

    return not _is_private_chat_type(getattr(getattr(message, "chat", None), "type", None))


def chat_owner_external_id(chat: Any, sender_id: int | None) -> str:
    """Backend ``owner_external_id`` for one Telegram chat participant.

    Private dialogs keep the legacy ``chat.id`` key.  In group chats each
    sender gets ``"<chat_id>:<sender_id>"`` so members do not share one
    history; the leading segment still parses as the Telegram delivery chat.
    """

    chat_id = int(chat.id)
    if _is_private_chat_type(getattr(chat, "type", None)):
        return str(chat_id)
    return f"{chat_id}:{sender_id if sender_id is not None else chat_id}"


def message_owner_external_id(message: Message) -> str:
    sender = getattr(message, "from_user", None) or getattr(message, "sender_chat", None)
    return chat_owner_external_id(message.chat, getattr(sender, "id", None))


def callback_owner_external_id(callback: CallbackQuery) -> str:
    """Owner key for callbacks: the chat plus the user who pressed the button."""

    message = getattr(callback, "message", None)
    chat = getattr(message, "chat", None)
    sender_id = getattr(getattr(callback, "from_user", None), "id", None)
    if chat is None:
        return str(sender_id if sender_id is not None else "")
    return chat_owner_external_id(chat, sender_id)


async def get_telegram_chat_id(message: Message, backend: BackendClient) -> UUID:
    return await backend.get_or_create_chat(message_owner_external_id(message), "telegram")


def message_text(message: Message) -> str:
    """Return message text/caption with hidden ``text_link`` URLs inlined.

    ``message.md_text`` would also resolve links, but it escapes everyday
    punctuation (``.``, ``!``, ``()``) which would pollute the content stored
    in chat history.  Only ``text_link`` entities carry data that is not
    already visible, so they are rewritten as ``[label](url)`` while the rest
    of the text stays untouched.  Telegram entity offsets are UTF-16 code
    units, so slicing happens on the surrogate-expanded representation.
    """

    text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
    entities = getattr(message, "entities", None) or getattr(message, "caption_entities", None) or []
    links = sorted(
        (entity for entity in entities if entity.type == "text_link" and entity.url),
        key=lambda entity: entity.offset,
    )
    if not links:
        return text.strip()
    raw = add_surrogates(text)
    pieces: list[bytes] = []
    cursor = 0
    for entity in links:
        start, end = entity.offset * 2, (entity.offset + entity.length) * 2
        if start < cursor or end > len(raw):
            continue
        pieces.append(raw[cursor:start])
        label = remove_surrogates(raw[start:end])
        pieces.append(add_surrogates(f"[{label}]({entity.url})"))
        cursor = end
    pieces.append(raw[cursor:])
    return remove_surrogates(b"".join(pieces)).strip()


def attachment_placeholder(message: Message) -> str | None:
    """Describe a non-text attachment the way the backend expects to see it."""

    if getattr(message, "photo", None):
        return "[фото]"
    if getattr(message, "voice", None):
        return "[голосовое сообщение]"
    if getattr(message, "audio", None):
        return "[аудиофайл]"
    if getattr(message, "video", None) or getattr(message, "video_note", None):
        return "[видео]"
    document = getattr(message, "document", None)
    if document is not None:
        return f"[документ {getattr(document, 'file_name', None) or ''}]".rstrip()
    if getattr(message, "sticker", None):
        return "[стикер]"
    if getattr(message, "location", None) or getattr(message, "venue", None):
        return "[геолокация]"
    if getattr(message, "contact", None):
        return "[контакт]"
    if getattr(message, "poll", None):
        return "[опрос]"
    return None


def quote_block(message: Message) -> str:
    """Render the message the user replied to as a context block for the LLM.

    Support engineers routinely answer by replying to the original client
    message; without this block the quoted text never reaches the backend.
    """

    reply = getattr(message, "reply_to_message", None)
    if reply is None:
        return ""
    quoted = message_text(reply) or attachment_placeholder(reply) or ""
    quoted = quoted.strip()
    if not quoted:
        return ""
    return f"[сообщение, на которое отвечает пользователь]:\n{quoted[:QUOTE_MAX_CHARS]}"


def compose_message_text(message: Message, *, placeholder: str = "") -> str:
    """Build the backend ``content`` for one incoming message."""

    parts = [quote_block(message), message_text(message) or placeholder]
    return "\n\n".join(part for part in parts if part).strip()


def is_bot_or_channel_edit(message: Message) -> bool:
    """Edits of our own streamed answers must not trigger a new request."""

    sender = getattr(message, "from_user", None)
    return sender is None or bool(getattr(sender, "is_bot", False))


def backend_error_message(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.ConnectError):
        return "Сервис недоступен, попробуйте позже."
    if isinstance(exc, httpx.ReadTimeout):
        return "Ответ занимает слишком долго. Попробуйте ещё раз позже."
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        error = _error_payload(exc.response)
        code = error.get("code")
        if status_code == 400 and code == "security_input_blocked":
            if error.get("rule") == "length":
                limit = error.get("limit")
                if isinstance(limit, int):
                    return f"Сообщение слишком длинное — максимум {limit} символов. Разделите его на несколько сообщений."
                return "Сообщение слишком длинное. Разделите его на несколько сообщений."
            return _SECURITY_RULE_MESSAGES.get(str(error.get("rule") or ""), _SECURITY_RULE_DEFAULT)
        if status_code == 403 and code == "moderation_blocked":
            return "Не могу обработать этот запрос — он мог нарушить правила."
        if code in _MEDIA_ERROR_MESSAGES:
            server_message = error.get("message")
            if code == "media_processing_failed" and isinstance(server_message, str) and server_message.strip():
                return server_message
            return _MEDIA_ERROR_MESSAGES[str(code)]
        if status_code == 429:
            return "Слишком много запросов, подождите минуту."
        if status_code >= 500:
            return "Внутренняя ошибка сервиса. Попробуйте позже."
    return "Не удалось обработать запрос. Проверьте данные и попробуйте ещё раз."


def _error_payload(response: httpx.Response) -> dict:
    try:
        payload = response.json()
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    error = payload.get("error", payload.get("detail", {}))
    return error if isinstance(error, dict) else {}


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
    return answer_kb(chat_id, message_id, source_document_count(backend.last_response_sources(chat_id)))


async def stream_to_chat(
    message: Message,
    tokens: AsyncIterator[str | StreamReplace],
    *,
    reply_markup_factory: Callable[[], InlineKeyboardMarkup | None] | None = None,
    sources_factory: Callable[[], list[dict]] | None = None,
    notice_factory: Callable[[], str] | None = None,
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
            if isinstance(delta, StreamReplace):
                # Canonical post-moderation text replaces the shown draft.
                buffer = delta.text
            else:
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
    if notice_factory is not None:
        notice = (notice_factory() or "").strip()
        if notice:
            final_text += f"\n\n{notice}"
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
        await send_answer(message, final_html, parse_mode=ParseMode.HTML)
    else:
        await send_answer(
            message,
            final_html,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )
    return buffer


render_stream = stream_to_chat


async def send_answer(message: Message, text: str, **kwargs):
    """Send an answer message; replies to the question in group chats.

    Group answers go through ``message.reply`` so members can attribute the
    bot's answer to the right question.  Private chats use ``message.answer``
    when present and fall back to ``bot.send_message`` for test transports.
    """

    method = "reply" if is_group_chat(message) else "answer"
    send = getattr(message, method, None)
    if callable(send):
        return await send(text, **kwargs)
    return await message.bot.send_message(chat_id=message.chat.id, text=text, **kwargs)


async def _start_editable_response(message: Message) -> Message | None:
    """Create one bot response to update in-place while SSE tokens arrive.

    The fallback intentionally keeps the old draft/final-message path for tests
    and for transports that cannot return an editable Telegram message.
    """

    send = getattr(message, "reply" if is_group_chat(message) else "answer", None)
    if not callable(send):
        return None
    try:
        return await send(RESPONSE_PLACEHOLDER)
    except TelegramAPIError:
        return None


def _format_cited_sources(answer: str, sources: list[dict]) -> str:
    source_numbers = _cited_source_numbers(answer, len(sources))
    if not source_numbers:
        return ""
    source_items = [(number, sources[number - 1]) for number in source_numbers]
    return _format_source_items(source_items, heading="Источники")


def format_all_sources(sources: list[dict]) -> str:
    return _format_source_items(list(enumerate(sources, start=1)), heading="Документы").removeprefix("\n\n")


def source_document_count(sources: list[dict]) -> int:
    return len(
        {
            str(source.get("file_name") or "источник").casefold()
            for source in sources
        }
    )


def _cited_source_numbers(answer: str, source_count: int) -> list[int]:
    cited: set[int] = set()
    for match in _CITATION_RE.finditer(answer):
        cited.update(int(value.strip()) for value in match.group(1).split(","))
    return [number for number in range(1, source_count + 1) if number in cited]


def _format_source_items(source_items: list[tuple[int, dict]], *, heading: str) -> str:
    if not source_items:
        return ""
    grouped: dict[str, dict[str, object]] = {}
    for index, source in source_items:
        file_name = str(source.get("file_name") or "источник")
        page = source.get("page")
        key = file_name.casefold()
        group = grouped.setdefault(
            key,
            {"file_name": file_name, "numbers": [], "page_pairs": []},
        )
        numbers = group["numbers"]
        page_pairs = group["page_pairs"]
        if isinstance(numbers, list):
            numbers.append(index)
        printed_page = source.get("printed_page")
        pair = (page, printed_page)
        if isinstance(page_pairs, list) and pair != (None, None) and pair not in page_pairs:
            page_pairs.append(pair)

    lines = []
    for group in grouped.values():
        numbers = group["numbers"]
        page_pairs = group["page_pairs"]
        citation = ", ".join(str(number) for number in numbers) if isinstance(numbers, list) else ""
        labels: list[str] = []
        if isinstance(page_pairs, list):
            for physical, printed in page_pairs:
                if physical is not None and printed is not None:
                    labels.append(f"PDF стр. {physical} (в документе {printed})")
                elif physical is not None:
                    labels.append(f"PDF стр. {physical}")
                elif printed is not None:
                    labels.append(f"стр. документа {printed}")
        page_text = f", {'; '.join(labels)}" if labels else ""
        lines.append(f"[{citation}] {group['file_name']}{page_text}")
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
