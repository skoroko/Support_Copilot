from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from io import BytesIO

from aiogram.types import Message

from app.observability.logging import get_logger
from bot.handlers._common import (
    BACKEND_EXCEPTIONS,
    backend_error_message,
    compose_message_text,
    message_owner_external_id,
    response_feedback_markup,
    send_answer,
    stream_to_chat,
)
from bot.services.backend_client import BackendClient

logger = get_logger("telegram-bot-burst")

TEXT_DEBOUNCE_SECONDS = 0.7
MEDIA_GROUP_DEBOUNCE_SECONDS = 1.5
MAX_BATCH_AGE_SECONDS = 4.0
MAX_BATCH_FILES = 10


@dataclass
class BurstItem:
    """One incoming Telegram message queued for a merged backend request."""

    message: Message
    media: tuple[str, str] | None = None  # (file_id, mime) for attachments
    placeholder: str = ""                 # text used when the attachment has no caption
    error: str | None = None              # pre-validation failure, reported at flush
    text: str = ""                        # quote + user text/caption (never the placeholder)


@dataclass
class _Batch:
    chat_id: int                       # Telegram chat for delivery/logging
    owner_key: str                     # backend owner key = _pending dict key
    items: list[BurstItem] = field(default_factory=list)
    by_message_id: dict[int, BurstItem] = field(default_factory=dict)
    anchor: Message | None = None
    first_seen: float = 0.0
    task: asyncio.Task | None = None


_pending: dict[str, _Batch] = {}


def enqueue_item(item: BurstItem, backend: BackendClient) -> None:
    _enqueue(message_owner_external_id(item.message), item.message.chat.id, backend, item)


def enqueue_text(message: Message, backend: BackendClient) -> None:
    enqueue_item(BurstItem(message=message, text=compose_message_text(message)), backend)


def absorb_edit(message: Message, backend: BackendClient, item: BurstItem) -> bool:
    """Fold an edited message into the pending batch.

    Returns True when the edit was queued: either it updates a message that is
    still waiting in the batch (replace in place), or no batch exists yet so
    the edit starts one and rapid re-edits coalesce.  Returns False when an
    unrelated batch is pending — the caller should deliver the edit on its own
    so a stale correction does not merge into a fresh question.
    """

    batch = _pending.get(message_owner_external_id(message))
    if batch is None:
        enqueue_item(item, backend)
        return True
    existing = batch.by_message_id.get(message.message_id)
    if existing is None:
        return False
    existing.message = message
    existing.media = item.media
    existing.placeholder = item.placeholder
    existing.error = item.error
    existing.text = item.text
    batch.anchor = message
    _reschedule(batch, backend)
    return True


async def deliver_now(item: BurstItem, backend: BackendClient) -> None:
    """Send a single item outside the batch (edit of an already-flushed message)."""

    batch = _Batch(
        chat_id=item.message.chat.id,
        owner_key=message_owner_external_id(item.message),
        items=[item],
        anchor=item.message,
    )
    await _deliver(batch, backend)


async def flush_pending(chat_id: int, backend: BackendClient, *, owner_key: str | None = None) -> None:
    """Deliver queued batches immediately (used by /clear to keep update order)."""

    keys = [key for key, batch in _pending.items()
            if batch.chat_id == chat_id and (owner_key is None or key == owner_key)]
    for key in keys:
        batch = _pending.pop(key, None)
        if batch is None:
            continue
        if batch.task is not None:
            batch.task.cancel()
        try:
            await _deliver(batch, backend)
        except Exception:
            logger.exception("burst_flush_failed", chat_id=chat_id)


def discard_pending(owner_key: str) -> None:
    """Drop only this owner's unsent batch when they start a new dialog."""
    batch = _pending.pop(owner_key, None)
    if batch is not None and batch.task is not None:
        batch.task.cancel()


def _enqueue(owner_key: str, chat_id: int, backend: BackendClient, item: BurstItem) -> None:
    batch = _pending.get(owner_key)
    if batch is None:
        batch = _Batch(chat_id=chat_id, owner_key=owner_key, first_seen=time.monotonic())
        _pending[owner_key] = batch
    batch.items.append(item)
    batch.by_message_id[item.message.message_id] = item
    batch.anchor = item.message
    _reschedule(batch, backend)


def _reschedule(batch: _Batch, backend: BackendClient) -> None:
    if batch.task is not None:
        batch.task.cancel()
    has_media_group = any(getattr(item.message, "media_group_id", None) for item in batch.items)
    delay = MEDIA_GROUP_DEBOUNCE_SECONDS if has_media_group else TEXT_DEBOUNCE_SECONDS
    remaining = batch.first_seen + MAX_BATCH_AGE_SECONDS - time.monotonic()
    delay = min(delay, max(remaining, 0.0))
    batch.task = asyncio.create_task(_flush_after(batch, delay, backend))


async def _flush_after(batch: _Batch, delay: float, backend: BackendClient) -> None:
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        return
    if _pending.get(batch.owner_key) is not batch:
        return
    del _pending[batch.owner_key]
    try:
        await _deliver(batch, backend)
    except Exception:
        logger.exception("burst_deliver_failed", chat_id=batch.chat_id)


async def _download(message: Message, file_id: str) -> bytes:
    telegram_file = await message.bot.get_file(file_id)
    if not telegram_file.file_path:
        raise ValueError("Telegram не вернул путь к файлу.")
    destination = BytesIO()
    await message.bot.download_file(telegram_file.file_path, destination=destination)
    return destination.getvalue()


async def _deliver(batch: _Batch, backend: BackendClient) -> None:
    anchor = batch.anchor
    if anchor is None:
        return
    slots: list[str] = []
    failures: list[str] = []
    downloads: list[tuple[int, BurstItem]] = []
    for item in batch.items:
        if item.media is None:
            slots.append(item.text)
            continue
        if item.error is not None:
            failures.append(item.error)
            slots.append(item.text)
            continue
        if len(downloads) >= MAX_BATCH_FILES:
            failures.append("слишком много вложений — часть файлов пропущена")
            slots.append(item.text)
            continue
        downloads.append((len(slots), item))
        slots.append("")

    files: list[tuple[bytes, str]] = []
    if downloads:
        results = await asyncio.gather(
            *(_download(item.message, item.media[0]) for _, item in downloads if item.media),
            return_exceptions=True,
        )
        for (slot, item), result in zip(downloads, results):
            if isinstance(result, BaseException):
                logger.warning(
                    "burst_media_download_failed",
                    chat_id=batch.chat_id,
                    error_type=type(result).__name__,
                )
                failures.append(f"не удалось загрузить {item.placeholder.strip('[]') or 'вложение'}")
                slots[slot] = item.text
                continue
            files.append((result, item.media[1]))
            slots[slot] = item.text or item.placeholder

    content = "\n".join(slot for slot in slots if slot and slot.strip()).strip()
    if not files and not content:
        if failures:
            await send_answer(anchor, "\n".join(dict.fromkeys(failures)))
        return
    if failures:
        await send_answer(anchor, "Часть вложений пропущена:\n" + "\n".join(dict.fromkeys(failures)))
    try:
        chat_id = await backend.get_or_create_chat(batch.owner_key, "telegram")
        await stream_to_chat(
            anchor,
            backend.send_message(chat_id, content or "[вложения]", media=files or None),
            reply_markup_factory=lambda: response_feedback_markup(backend, chat_id),
            sources_factory=lambda: backend.last_response_sources(chat_id),
            notice_factory=lambda: backend.last_response_notice(chat_id),
        )
    except BACKEND_EXCEPTIONS as exc:
        await send_answer(anchor, backend_error_message(exc))
