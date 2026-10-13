from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest
from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramNetworkError
from aiogram.filters import Command
from aiogram.methods import AnswerCallbackQuery, EditMessageText, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from bot.handlers import menu
from bot.handlers._burst import BurstItem, _pending, discard_pending, enqueue_item
from bot.handlers.commands import clear, start
from bot.keyboards.menu import home_kb, menu_data, parse_menu_data
from bot.navigation import register_menu
from bot.services.backend_client import BackendClient


class Harness:
    def __init__(self):
        self.methods = []
        self.session = AsyncMock(side_effect=self.request)
        self.bot = Bot("123:TEST", session=self.session)
        self.backend = AsyncMock(spec=BackendClient)
        self.backend.get_or_create_chat.return_value = uuid4()
        self.backend.list_messages.return_value = []
        self.backend.pending_approval.return_value = None
        self.dispatcher = Dispatcher()
        router = Router()
        router.message.register(menu.menu_command, Command("menu"))
        router.message.register(start, Command("start"))
        router.message.register(clear, Command("clear"))
        router.callback_query.register(menu.navigate, F.data.startswith("mn:"))
        self.dispatcher.include_router(router)

    async def request(self, bot, method, **kwargs):
        self.methods.append(method)
        return True

    def message(self, *, chat_id=-100, sender=10, text=None):
        return Message(message_id=7, date=datetime.now(UTC),
                       chat=Chat(id=chat_id, type="supergroup" if chat_id < 0 else "private"),
                       from_user=User(id=sender, is_bot=False, first_name="User"), text=text)

    async def click(self, action, *, sender=10, data=None, message=True):
        callback = CallbackQuery(id="click", from_user=User(id=sender, is_bot=False, first_name="User"),
                                 chat_instance="chat", data=data or menu_data(10, action),
                                 message=self.message(sender=123) if message else None)
        await self.dispatcher.feed_update(self.bot, Update(update_id=1, callback_query=callback), backend=self.backend)

    def last_text(self):
        return next(method for method in reversed(self.methods) if isinstance(method, (EditMessageText, SendMessage)))


@pytest.mark.asyncio
async def test_native_registration_and_temporary_failure_are_best_effort():
    bot = AsyncMock()
    await register_menu(bot)
    assert bot.set_chat_menu_button.await_args.kwargs["menu_button"].type == "commands"
    assert "menu" in [command.command for command in bot.set_my_commands.await_args.args[0]]
    bot.set_my_commands.side_effect = TelegramNetworkError(method="setMyCommands", message="offline")
    await register_menu(bot)  # Polling may proceed even when registration fails.


@pytest.mark.asyncio
async def test_menu_command_runs_through_dispatcher_without_backend_access():
    h = Harness()
    await h.dispatcher.feed_update(h.bot, Update(update_id=1, message=h.message(text="/menu")), backend=h.backend)
    assert h.last_text().reply_markup == home_kb(10)
    assert h.backend.mock_calls == []
    assert all(len(button.callback_data.encode()) <= 64 for row in home_kb(10).inline_keyboard for button in row)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["examples", "help", "home"])
async def test_navigation_does_not_submit_or_mutate_history(action):
    h = Harness()
    await h.click(action)
    assert isinstance(h.methods[0], AnswerCallbackQuery)
    assert h.backend.mock_calls == []
    assert h.last_text().reply_markup


@pytest.mark.asyncio
async def test_history_owner_comes_from_click_and_ack_precedes_backend_read():
    h = Harness()

    async def get_chat(owner, channel):
        assert isinstance(h.methods[0], AnswerCallbackQuery)
        assert owner == "-100:10"  # Callback's bot-authored message is not the owner.
        return uuid4()

    h.backend.get_or_create_chat.side_effect = get_chat
    h.backend.list_messages.return_value = [{"role": "user", "content": "<мой вопрос>"}]
    await h.click("history")
    assert "&lt;мой вопрос&gt;" in h.last_text().text
    h.backend.send_message.assert_not_called()
    h.backend.clear_messages.assert_not_called()


@pytest.mark.asyncio
async def test_pending_panel_preserves_exact_text_and_existing_approval_callbacks():
    h = Harness()
    h.backend.pending_approval.return_value = {"approval_id": str(uuid4()), "chat_id": -555,
                                               "text": "**буквально** `<текст>` & [1]"}
    await h.click("pending")
    result = h.last_text()
    assert "-555" in result.text
    assert "**буквально** `&lt;текст&gt;` &amp; [1]" in result.text
    callbacks = [button.callback_data for row in result.reply_markup.inline_keyboard for button in row]
    assert callbacks[0].startswith("ap:y:") and callbacks[1].startswith("ap:n:")
    assert callbacks[2] == menu_data(10, "home")
    h.backend.resume_approval.assert_not_called()


@pytest.mark.asyncio
async def test_start_restores_exact_durable_pending_preview():
    h = Harness()
    h.backend.pending_approval.return_value = {"approval_id": str(uuid4()), "chat_id": 123,
                                               "text": "**готовый текст**"}
    await h.dispatcher.feed_update(h.bot, Update(update_id=1, message=h.message(text="/start")), backend=h.backend)
    assert "**готовый текст**" in h.last_text().text
    h.backend.pending_approval.assert_awaited_once()


@pytest.mark.asyncio
async def test_reset_requires_confirmation_cancellation_preserves_state_and_replay_is_harmless():
    h = Harness()
    await h.click("new")
    old_data = h.last_text().reply_markup.inline_keyboard[0][0].callback_data
    h.backend.clear_messages.assert_not_called()
    await h.click("home")
    await h.click("reset", data=old_data)
    h.backend.clear_messages.assert_not_called()
    await h.click("new")
    data = h.last_text().reply_markup.inline_keyboard[0][0].callback_data
    with patch("bot.handlers.menu.discard_pending") as discard:
        await h.click("reset", data=data)
        discard.assert_called_once_with("-100:10")
    h.backend.clear_messages.assert_awaited_once()
    await h.click("reset", data=data)
    h.backend.clear_messages.assert_awaited_once()
    h.backend.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_old_confirmation_does_not_consume_newer_confirmation():
    h = Harness()
    await h.click("new")
    old_data = h.last_text().reply_markup.inline_keyboard[0][0].callback_data
    await h.click("new")
    new_data = h.last_text().reply_markup.inline_keyboard[0][0].callback_data
    await h.click("reset", data=old_data)
    h.backend.clear_messages.assert_not_called()
    await h.click("reset", data=new_data)
    h.backend.clear_messages.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/menu", "/start", "/clear"])
async def test_home_and_clear_commands_invalidate_old_reset_buttons(command):
    h = Harness()
    await h.click("new")
    old_data = h.last_text().reply_markup.inline_keyboard[0][0].callback_data
    await h.dispatcher.feed_update(
        h.bot, Update(update_id=2, message=h.message(text=command)), backend=h.backend
    )
    previous_clears = h.backend.clear_messages.await_count
    await h.click("reset", data=old_data)
    assert h.backend.clear_messages.await_count == previous_clears
    assert "устарело" in h.last_text().text


@pytest.mark.asyncio
async def test_expired_confirmation_does_not_clear():
    h = Harness()
    await h.click("new")
    data = h.last_text().reply_markup.inline_keyboard[0][0].callback_data
    with patch("bot.handlers.menu.time.monotonic", return_value=float("inf")):
        await h.click("reset", data=data)
    h.backend.clear_messages.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("data", ["mn:2:10:history", "mn:1:11:history", "mn:1:10:history"])
async def test_expired_telegram_callback_does_not_raise(data):
    h = Harness()
    h.session.side_effect = TelegramNetworkError(method="answerCallbackQuery", message="expired")
    await h.click("history", data=data)
    assert h.backend.mock_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("data,sender,message", [
    (menu_data(10, "history"), 11, True), ("mn:2:10:reset", 10, True),
    ("mn:1:10:unknown", 10, True), ("mn:1:10:reset", 10, True),
    (menu_data(10, "history"), 10, False),
])
async def test_foreign_invalid_and_unavailable_buttons_do_not_access_backend(data, sender, message):
    h = Harness()
    await h.click("history", data=data, sender=sender, message=message)
    assert h.backend.mock_calls == []
    assert isinstance(h.methods[0], AnswerCallbackQuery)
    assert h.methods[0].show_alert


@pytest.mark.asyncio
async def test_backend_error_is_acknowledged_and_readable():
    h = Harness()
    h.backend.list_messages.side_effect = httpx.ConnectError("offline")
    await h.click("history")
    assert isinstance(h.methods[0], AnswerCallbackQuery)
    assert "недоступен" in h.last_text().text


@pytest.mark.asyncio
async def test_discard_pending_does_not_flush_another_group_member():
    h = Harness()
    try:
        enqueue_item(BurstItem(message=h.message(sender=10), text="one"), h.backend)
        enqueue_item(BurstItem(message=h.message(sender=11), text="two"), h.backend)
        discard_pending("-100:10")
        assert "-100:10" not in _pending and "-100:11" in _pending
        h.backend.send_message.assert_not_called()
    finally:
        discard_pending("-100:10")
        discard_pending("-100:11")


def test_parser_rejects_extra_segments():
    with pytest.raises(ValueError):
        parse_menu_data("mn:1:10:home:abcdefghijkl")
