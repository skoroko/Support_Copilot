from __future__ import annotations

from aiogram import F, Router
from aiogram.types import Message

from bot.handlers import _burst
from bot.handlers._common import compose_message_text, is_bot_or_channel_edit
from bot.services.backend_client import BackendClient

router = Router()
MAX_PHOTO_BYTES = 2 * 1024 * 1024
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _media_descriptor(message: Message) -> tuple[str, str, str, str | None] | None:
    """Return (file_id, mime, placeholder, validation_error) for the attachment."""

    if message.photo:
        candidates = [photo for photo in message.photo if (photo.file_size or 0) <= MAX_PHOTO_BYTES]
        if not candidates:
            return "", "", "[фото]", "фото слишком большое — максимальный размер 2 МБ"
        photo = max(candidates, key=lambda item: item.file_size or 0)
        return photo.file_id, "image/jpeg", "[фото]", None
    if message.voice:
        return message.voice.file_id, "audio/ogg", "[голосовое сообщение]", None
    if message.audio:
        return (
            message.audio.file_id,
            message.audio.mime_type or "audio/mpeg",
            "[аудиофайл]",
            None,
        )
    if message.document:
        document = message.document
        filename = document.file_name or ""
        if not filename.lower().endswith((".pdf", ".docx")):
            return "", "", f"[документ {filename}]".rstrip(), (
                f"документ {filename or ''} не поддерживается — только PDF и DOCX".rstrip()
            )
        if (document.file_size or 0) > MAX_DOCUMENT_BYTES:
            return "", "", f"[документ {filename}]".rstrip(), (
                f"документ {filename or ''} слишком большой — максимальный размер 10 МБ".rstrip()
            )
        mime = document.mime_type or ("application/pdf" if filename.lower().endswith(".pdf") else DOCX_MIME)
        return document.file_id, mime, f"[документ {filename}]".rstrip(), None
    return None


def _build_item(message: Message) -> _burst.BurstItem | None:
    descriptor = _media_descriptor(message)
    if descriptor is None:
        return None
    file_id, mime, placeholder, error = descriptor
    return _burst.BurstItem(
        message=message,
        media=(file_id, mime),
        placeholder=placeholder,
        error=error,
        text=compose_message_text(message),
    )


@router.message(F.photo | F.voice | F.audio | F.document)
async def handle_media(message: Message, backend: BackendClient) -> None:
    item = _build_item(message)
    if item is not None:
        _burst.enqueue_item(item, backend)


@router.edited_message(F.photo | F.voice | F.audio | F.document)
async def handle_edited_media(message: Message, backend: BackendClient) -> None:
    if is_bot_or_channel_edit(message):
        return
    item = _build_item(message)
    if item is None:
        return
    if _burst.absorb_edit(message, backend, item):
        return
    await _burst.deliver_now(item, backend)
