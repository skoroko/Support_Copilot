from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.handlers._burst import flush_pending
from bot.handlers._common import (
    BACKEND_EXCEPTIONS,
    backend_error_message,
    get_telegram_chat_id,
    message_owner_external_id,
    send_rendered_answer,
)
from bot.handlers.menu import HELP, HOME, cancel_reset_confirmation, render_pending
from bot.keyboards.menu import back_kb, home_kb
from bot.services.backend_client import BackendClient

router = Router()


@router.message(Command("start"))
async def start(message: Message, backend: BackendClient) -> None:
    cancel_reset_confirmation(message)
    try:
        chat_id = await get_telegram_chat_id(message, backend)
    except BACKEND_EXCEPTIONS as exc:
        await message.answer(backend_error_message(exc))
        return

    if message.from_user:
        await send_rendered_answer(message, HOME, reply_markup=home_kb(message.from_user.id))
    if isinstance(backend, BackendClient):
        try:
            draft = await backend.pending_approval(chat_id)
            if draft and message.from_user:
                await render_pending(message, chat_id, message.from_user.id, draft)
        except BACKEND_EXCEPTIONS as exc:
            await message.answer(backend_error_message(exc))


@router.message(Command("help"))
async def help_command(message: Message) -> None:
    await send_rendered_answer(message, HELP, reply_markup=back_kb(message.from_user.id) if message.from_user else None)


@router.message(Command("clear"))
async def clear(message: Message, backend: BackendClient) -> None:
    cancel_reset_confirmation(message)
    # A debounced question must reach the backend before its history is wiped.
    await flush_pending(message.chat.id, backend, owner_key=message_owner_external_id(message))
    try:
        chat_id = await get_telegram_chat_id(message, backend)
        await backend.clear_messages(chat_id)
    except BACKEND_EXCEPTIONS as exc:
        await message.answer(backend_error_message(exc))
        return

    await message.answer("История очищена.")


@router.message(Command("history"))
async def history(message: Message, backend: BackendClient) -> None:
    try:
        chat_id = await get_telegram_chat_id(message, backend)
        messages = await backend.list_messages(chat_id)
    except BACKEND_EXCEPTIONS as exc:
        await message.answer(backend_error_message(exc))
        return
    text = "\n\n".join(f"{'Вы' if item.get('role') == 'user' else 'Ассистент'}: {item.get('content', '')}" for item in messages)
    await send_rendered_answer(message, text if text else "История пуста.")
