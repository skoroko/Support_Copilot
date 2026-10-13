from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.handlers._burst import flush_pending
from bot.handlers._common import (
    BACKEND_EXCEPTIONS,
    backend_error_message,
    get_telegram_chat_id,
    send_rendered_answer,
)
from bot.keyboards.inline import approval_kb
from bot.services.backend_client import BackendClient

router = Router()


@router.message(Command("start"))
async def start(message: Message, backend: BackendClient) -> None:
    try:
        chat_id = await get_telegram_chat_id(message, backend)
    except BACKEND_EXCEPTIONS as exc:
        await message.answer(backend_error_message(exc))
        return

    await message.answer(
        "Support Copilot подключен.\n\n"
        "Пишите вопрос обычным сообщением, используйте "
        "/clear для очистки истории и /help для списка команд."
    )
    if isinstance(backend, BackendClient):
        try:
            draft = await backend.pending_approval(chat_id)
            if draft:
                await send_rendered_answer(message, backend.last_response_notice(chat_id),
                                           reply_markup=approval_kb(chat_id, draft["approval_id"]))
        except BACKEND_EXCEPTIONS as exc:
            await message.answer(backend_error_message(exc))


@router.message(Command("help"))
async def help_command(message: Message) -> None:
    await message.answer(
        "Команды:\n"
        "/start - создать чат и начать работу\n"
        "/clear - очистить историю текущего чата\n"
        "/history - последние сообщения\n"
        "/help - показать эту справку"
    )


@router.message(Command("clear"))
async def clear(message: Message, backend: BackendClient) -> None:
    # A debounced question must reach the backend before its history is wiped.
    await flush_pending(message.chat.id, backend)
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
