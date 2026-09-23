from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.handlers._burst import flush_pending
from bot.handlers._common import BACKEND_EXCEPTIONS, backend_error_message, get_telegram_chat_id
from bot.services.backend_client import BackendClient


router = Router()


@router.message(Command("cancel"))
async def cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Сценарий отменен.")


@router.message(Command("start"))
async def start(message: Message, backend: BackendClient) -> None:
    try:
        await get_telegram_chat_id(message, backend)
    except BACKEND_EXCEPTIONS as exc:
        await message.answer(backend_error_message(exc))
        return

    await message.answer(
        "Support Copilot подключен.\n\n"
        "Пишите вопрос обычным сообщением, используйте /ask для сценария с темой, "
        "/clear для очистки истории и /help для списка команд."
    )


@router.message(Command("help"))
async def help_command(message: Message) -> None:
    await message.answer(
        "Команды:\n"
        "/start - создать чат и начать работу\n"
        "/ask - задать вопрос через выбор темы\n"
        "/clear - очистить историю текущего чата\n"
        "/cancel - отменить активный сценарий\n"
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
