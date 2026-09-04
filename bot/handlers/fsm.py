from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.handlers._common import BACKEND_EXCEPTIONS, backend_error_message, get_telegram_chat_id, stream_to_chat
from bot.keyboards.inline import SUPPORT_TOPICS, topics_kb
from bot.services.backend_client import BackendClient
from bot.states import AskFlow


router = Router()


@router.message(Command("ask"))
async def start_ask(message: Message, state: FSMContext) -> None:
    await state.set_state(AskFlow.waiting_for_topic)
    await message.answer("Выберите тему обращения:", reply_markup=topics_kb())


@router.callback_query(AskFlow.waiting_for_topic, F.data.startswith("topic:"))
async def choose_topic(callback: CallbackQuery, state: FSMContext) -> None:
    data = callback.data or ""
    slug = data.split(":", maxsplit=1)[1] if ":" in data else ""
    if slug == "cancel":
        await state.clear()
        if callback.message:
            await callback.message.edit_text("Сценарий отменен.")
        await callback.answer()
        return

    topic = SUPPORT_TOPICS.get(slug, slug)
    await state.update_data(topic=topic)
    await state.set_state(AskFlow.waiting_for_question)
    if callback.message:
        await callback.message.edit_text(f"Тема: {topic}\nНапишите вопрос одним сообщением.")
    await callback.answer()


@router.message(AskFlow.waiting_for_question, F.text)
async def ask_question(message: Message, state: FSMContext, backend: BackendClient) -> None:
    data = await state.get_data()
    topic = str(data.get("topic") or "Поддержка")
    prompt = f"Тема: {topic}. Вопрос: {message.text or ''}"

    try:
        chat_id = await get_telegram_chat_id(message, backend)
        await stream_to_chat(message, backend.send_message(chat_id, prompt))
    except BACKEND_EXCEPTIONS as exc:
        await message.answer(backend_error_message(exc))
    finally:
        await state.clear()
