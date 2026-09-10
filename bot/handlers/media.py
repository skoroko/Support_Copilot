from __future__ import annotations

from io import BytesIO

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
MAX_PHOTO_BYTES = 2 * 1024 * 1024
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


async def _download(message: Message, file_id: str) -> bytes:
    telegram_file = await message.bot.get_file(file_id)
    if not telegram_file.file_path:
        raise ValueError("Telegram не вернул путь к файлу.")
    destination = BytesIO()
    await message.bot.download_file(telegram_file.file_path, destination=destination)
    return destination.getvalue()


async def _send_media(
    message: Message,
    backend: BackendClient,
    *,
    file_id: str,
    content: str,
    mime: str,
) -> None:
    try:
        media = await _download(message, file_id)
        chat_id = await get_telegram_chat_id(message, backend)
        await stream_to_chat(
            message,
            backend.send_message(chat_id, content, media=media, mime=mime),
            reply_markup_factory=lambda: response_feedback_markup(backend, chat_id),
            sources_factory=lambda: backend.last_response_sources(chat_id),
        )
    except BACKEND_EXCEPTIONS as exc:
        await message.answer(backend_error_message(exc))


@router.message(F.photo)
async def handle_photo(message: Message, backend: BackendClient) -> None:
    candidates = [photo for photo in message.photo if (photo.file_size or 0) <= MAX_PHOTO_BYTES]
    if not candidates:
        await message.answer("Фото слишком большое. Максимальный размер — 2 МБ.")
        return
    photo = max(candidates, key=lambda item: item.file_size or 0)
    await _send_media(
        message,
        backend,
        file_id=photo.file_id,
        content=message.caption or "[фото]",
        mime="image/jpeg",
    )


@router.message(F.voice)
async def handle_voice(message: Message, backend: BackendClient) -> None:
    await _send_media(
        message,
        backend,
        file_id=message.voice.file_id,
        content=message.caption or "[голосовое сообщение]",
        mime="audio/ogg",
    )


@router.message(F.audio)
async def handle_audio(message: Message, backend: BackendClient) -> None:
    await _send_media(
        message,
        backend,
        file_id=message.audio.file_id,
        content=message.caption or "[аудиофайл]",
        mime=message.audio.mime_type or "audio/mpeg",
    )


@router.message(F.document)
async def handle_document(message: Message, backend: BackendClient) -> None:
    document = message.document
    filename = (document.file_name or "").lower()
    if not filename.endswith((".pdf", ".docx")):
        await message.answer("Поддерживаются только документы PDF и DOCX.")
        return
    if (document.file_size or 0) > MAX_DOCUMENT_BYTES:
        await message.answer("Документ слишком большой. Максимальный размер — 10 МБ.")
        return
    mime = document.mime_type or ("application/pdf" if filename.endswith(".pdf") else DOCX_MIME)
    await _send_media(
        message,
        backend,
        file_id=document.file_id,
        content=message.caption or f"[документ {document.file_name or ''}]",
        mime=mime,
    )
