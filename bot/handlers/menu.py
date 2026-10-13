"""Navigation never submits a question or executes a pending send."""

import secrets
import time
from uuid import UUID

from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from bot.handlers._burst import discard_pending
from bot.handlers._common import (
    BACKEND_EXCEPTIONS,
    backend_error_message,
    callback_owner_external_id,
    send_rendered_answer,
)
from bot.keyboards.inline import approval_kb
from bot.keyboards.menu import back_kb, home_kb, parse_menu_data, reset_kb
from bot.services.backend_client import BackendClient

router = Router()
HOME = (
    "Support Copilot\n\n"
    "Помогу найти ответ в руководствах и подготовить ответ пользователю. "
    "Напишите вопрос сообщением или выберите раздел ниже.\n\n"
    "Отправка подготовленного сообщения требует вашего подтверждения."
)
HELP = (
    "Как пользоваться\n\n"
    "1. Напишите вопрос и укажите продукт, версию и детали задачи. "
    "Можно приложить фото, PDF/DOCX или голосовое сообщение. К фото добавляйте текстовое описание.\n"
    "2. Уточняйте ответ в том же диалоге. Проверяйте документы по кнопке «Документы» под ответом.\n"
    "3. Для черновика напишите: «Подготовь ответ пользователю: …». "
    "Для отправки укажите получателя и текст, затем проверьте точный текст перед подтверждением.\n\n"
    "/menu — меню · /history — история · /help — справка\n"
    "/clear — сразу очистить историю и отменить ожидающую отправку."
)
EXAMPLES = (
    "Примеры вопросов\n\n"
    "• Какие граничные условия описаны в руководстве GTS NX?\n"
    "• Найди в руководстве GTS NX описание Mesh Generate.\n"
    "• Подготовь ответ пользователю по этому фрагменту руководства: …\n\n"
    "Скопируйте подходящий пример в сообщение и добавьте детали. "
    "Если в документах нет ответа, потребуется уточнение или помощь специалиста."
)
_CONFIRM_SECONDS = 300.0
_confirmations: dict[tuple[int, int], tuple[str, float]] = {}


def cancel_reset_confirmation(message: Message) -> None:
    """A refreshed or cleared conversation cannot reuse an old reset button."""

    if message.from_user:
        _confirmations.pop((message.chat.id, message.from_user.id), None)


async def _acknowledge(callback: CallbackQuery, text: str | None = None, *, show_alert: bool = False) -> bool:
    try:
        await callback.answer(text, show_alert=show_alert)
        return True
    except TelegramAPIError:
        return False


def history_text(messages: list[dict]) -> str:
    text = "\n\n".join(
        f"{'Вы' if item.get('role') == 'user' else 'Ассистент'}: {item.get('content', '')}"
        for item in messages
    )
    return "Последние сообщения\n\n" + text if text else "История пуста."


async def show_pending(message: Message, backend: BackendClient, chat_id: UUID, user_id: int, *, editable=None) -> None:
    draft = await backend.pending_approval(chat_id)
    await render_pending(message, chat_id, user_id, draft, editable=editable)


async def render_pending(message: Message, chat_id: UUID, user_id: int, draft: dict | None, *, editable=None) -> None:
    markup = back_kb(user_id)
    text = "Ожидающих отправки сообщений нет."
    if draft:
        # Read from the durable draft, not an earlier cached response notice.
        text = f"Подтвердите отправку.\nПолучатель (chat_id): {draft['chat_id']}\nТочный текст:\n{draft['text']}"
        markup = approval_kb(chat_id, draft["approval_id"])
        markup.inline_keyboard.extend(back_kb(user_id).inline_keyboard)
    await send_rendered_answer(message, text, reply_markup=markup, editable=editable)


@router.message(Command("menu"))
async def menu_command(message: Message) -> None:
    cancel_reset_confirmation(message)
    if message.from_user:
        await send_rendered_answer(message, HOME, reply_markup=home_kb(message.from_user.id))


@router.callback_query(F.data.startswith("mn:"))
async def navigate(callback: CallbackQuery, backend: BackendClient) -> None:
    try:
        user_id, action, token = parse_menu_data(callback.data or "")
    except ValueError:
        await _acknowledge(callback, "Это меню устарело. Откройте /menu.", show_alert=True)
        return
    if user_id != callback.from_user.id:
        await _acknowledge(callback, "Это меню другого пользователя. Откройте своё через /menu.", show_alert=True)
        return
    if not isinstance(callback.message, Message):
        await _acknowledge(callback, "Сообщение недоступно. Откройте /menu.", show_alert=True)
        return
    # Acknowledge before any backend I/O; expired Telegram callbacks are harmless.
    if not await _acknowledge(callback):
        return
    message = callback.message
    key = (message.chat.id, user_id)
    try:
        markup = back_kb(user_id)
        if action == "home":
            _confirmations.pop(key, None)
            text, markup = HOME, home_kb(user_id)
        elif action in {"examples", "help"}:
            text = EXAMPLES if action == "examples" else HELP
        elif action == "new":
            now = time.monotonic()
            for old_key, (_, expires) in list(_confirmations.items()):
                if expires <= now:
                    del _confirmations[old_key]
            if len(_confirmations) >= 1024:
                del _confirmations[next(iter(_confirmations))]
            nonce = secrets.token_urlsafe(9)
            _confirmations[key] = (nonce, now + _CONFIRM_SECONDS)
            text = ("Начать новый диалог?\n\nТекущая история и ожидающая отправка будут удалены. "
                    "Это действие нельзя отменить.")
            markup = reset_kb(user_id, nonce)
        else:
            if action == "reset":
                confirmation = _confirmations.get(key)
                if not confirmation or confirmation[0] != token or confirmation[1] <= time.monotonic():
                    await send_rendered_answer(message, "Подтверждение устарело. Откройте меню и повторите действие.",
                                               reply_markup=home_kb(user_id), editable=message)
                    return
                _confirmations.pop(key, None)
            owner = callback_owner_external_id(callback)
            chat_id = await backend.get_or_create_chat(owner, "telegram")
            if action == "pending":
                await show_pending(message, backend, chat_id, user_id, editable=message)
                return
            if action == "history":
                text = history_text(await backend.list_messages(chat_id))
            else:
                discard_pending(owner)
                await backend.clear_messages(chat_id)
                text, markup = "Новый диалог начат. Напишите вопрос.", home_kb(user_id)
        await send_rendered_answer(message, text, reply_markup=markup, editable=message)
    except BACKEND_EXCEPTIONS as exc:
        try:
            await send_rendered_answer(message, backend_error_message(exc), reply_markup=back_kb(user_id))
        except TelegramAPIError:
            return
    except TelegramAPIError:
        # Deleted/unchanged messages and failed delivery must not stop polling.
        return
