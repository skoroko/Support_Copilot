import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import (
    AnswerCallbackQuery,
    DeleteMessage,
    EditMessageReplyMarkup,
    EditMessageText,
    SendChatAction,
    SendMessage,
)
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from bot.handlers.expansion import COST_NOTICE, UNAVAILABLE, expansion
from bot.handlers.feedback import save_feedback
from bot.keyboards.inline import answer_kb, encode_message_context, expansion_kb
from bot.services.backend_client import BackendClient


class Harness:
    def __init__(self):
        self.methods = []
        self.fail_edit = False
        self.next_message_id = 100
        self.messages = {}
        self.typing = False
        self.typing_started = asyncio.Event()
        self.typing_release = None
        self.typing_finished = asyncio.Event()
        self.typing_tasks = []
        self.chat_type, self.telegram_chat_id = "supergroup", -100
        self.backend = AsyncMock(spec=BackendClient)
        self.chat_id, self.message_id = uuid4(), uuid4()
        self.backend.preview_expansion.return_value = {"available": True, "max_output_tokens": 1500}
        self.backend.expand_response.return_value = {"answer": "Подробный ответ.", "sources": [], "message_id": str(uuid4())}
        self.bot = Bot("123:TEST", session=AsyncMock(side_effect=self.request))
        self.dispatcher = Dispatcher()
        router = Router()
        router.callback_query.register(expansion, F.data.startswith("exp:"))
        router.callback_query.register(save_feedback, F.data.startswith("fb:"))
        self.dispatcher.include_router(router)

    async def request(self, bot, method, **kwargs):
        self.methods.append(method)
        if self.fail_edit and isinstance(method, EditMessageReplyMarkup):
            raise TelegramBadRequest(method=method, message="not editable")
        if isinstance(method, SendChatAction):
            self.typing_tasks.append(asyncio.current_task())
            self.typing = True
            self.typing_started.set()
            if self.typing_release is not None:
                await self.typing_release.wait()
            self.typing_finished.set()
        if isinstance(method, SendMessage):
            self.typing = False  # Telegram guarantees clearing only on a new message.
            self.next_message_id += 1
            self.messages[self.next_message_id] = {"text": method.text, "markup": method.reply_markup}
            return Message(message_id=self.next_message_id, date=datetime.now(UTC),
                           chat=Chat(id=method.chat_id, type=self.chat_type), text=method.text).as_(bot)
        if isinstance(method, EditMessageText):
            self.messages[method.message_id] = {"text": method.text, "markup": method.reply_markup}
        if isinstance(method, DeleteMessage):
            self.messages.pop(method.message_id, None)
        return True

    async def click(self, action, *, sender=7, data=None, markup=None):
        message = Message(message_id=4, date=datetime.now(UTC), chat=Chat(id=self.telegram_chat_id, type=self.chat_type),
                          from_user=User(id=123, first_name="Bot", is_bot=True), reply_markup=markup)
        callback = CallbackQuery(id="click", chat_instance="group", message=message,
            from_user=User(id=sender, first_name="User", is_bot=False),
            data=data or f"exp:{action}:{encode_message_context(self.chat_id, self.message_id)}")
        await self.dispatcher.feed_update(self.bot, Update(update_id=1, callback_query=callback), backend=self.backend)


@pytest.mark.asyncio
async def test_dispatcher_preview_and_cancel_do_not_generate():
    h = Harness()
    await h.click("a")
    assert isinstance(h.methods[0], AnswerCallbackQuery)
    panel = next(method for method in h.methods if isinstance(method, SendMessage))
    assert panel.text == COST_NOTICE
    assert [b.text for row in panel.reply_markup.inline_keyboard for b in row] == ["Подготовить подробный ответ", "Оставить краткий"]
    await h.click("n")
    assert h.backend.preview_expansion.await_count == 2
    assert h.backend.preview_expansion.await_args.kwargs["owner_external_id"] == "-100:7"
    h.backend.expand_response.assert_not_called()
    assert not any(isinstance(method, SendChatAction) for method in h.methods)
    assert not any(item["text"] == "Готовлю ответ…" for item in h.messages.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["a", "n", "y"])
async def test_dispatcher_foreign_group_callback_fails_without_edit_or_generation(action):
    h = Harness()
    request = httpx.Request("GET", "http://backend/preview")
    error = httpx.HTTPStatusError("not owned", request=request, response=httpx.Response(404, request=request))
    h.backend.preview_expansion.side_effect = error
    h.backend.expand_response.side_effect = error
    await h.click(action, sender=8)
    assert not any(isinstance(method, EditMessageReplyMarkup) for method in h.methods)
    assert any(isinstance(method, SendMessage) and method.text == UNAVAILABLE for method in h.methods)
    called = h.backend.expand_response if action == "y" else h.backend.preview_expansion
    assert called.await_args.kwargs["owner_external_id"] == "-100:8"


@pytest.mark.asyncio
async def test_dispatcher_confirm_delivers_saved_result_even_if_panel_edit_fails():
    h = Harness()
    h.fail_edit = True
    await h.click("y")
    h.backend.expand_response.assert_awaited_once_with(h.chat_id, h.message_id, owner_external_id="-100:7")
    response = next(iter(h.messages.values()))
    assert response["text"] == "Подробный ответ."
    assert not any((b.callback_data or "").startswith("exp:") for row in response["markup"].inline_keyboard for b in row)


@pytest.mark.asyncio
async def test_vote_keeps_exact_documents_and_expansion_affordance():
    h = Harness()
    h.backend.get_message_sources.return_value = [{"file_name": "Manual.pdf"}]
    markup = answer_kb(h.chat_id, h.message_id, 1, expandable=True)
    await h.click("", data=f"fb:u:{encode_message_context(h.chat_id, h.message_id)}", markup=markup)
    edit = next(method for method in h.methods if isinstance(method, EditMessageReplyMarkup))
    assert [b.text for row in edit.reply_markup.inline_keyboard for b in row] == ["📚 Документы (1)", "Подробнее"]


def test_all_expansion_callbacks_fit_telegram_limit():
    chat_id, message_id = uuid4(), uuid4()
    for keyboard in (answer_kb(chat_id, message_id, 99, expandable=True), expansion_kb(chat_id, message_id)):
        assert all(len(b.callback_data.encode()) <= 64 for row in keyboard.inline_keyboard for b in row)


@pytest.mark.asyncio
async def test_expansion_http_mutation_has_no_automatic_connect_retry():
    calls = []

    def fail(request):
        calls.append(request)
        assert request.headers["X-User-ID"] == "-100:8"
        assert request.content == b'{"confirmed":true}'
        raise httpx.ConnectError("offline", request=request)

    async with httpx.AsyncClient(base_url="http://backend", transport=httpx.MockTransport(fail)) as http:
        backend = BackendClient("http://backend", client=http, internal_token="offline-test")
        with pytest.raises(httpx.ConnectError):
            await backend.expand_response(uuid4(), uuid4(), owner_external_id="-100:8")
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("group", [False, True], ids=["private", "group"])
@pytest.mark.parametrize("error", [False, True], ids=["saved-result", "unavailable"])
async def test_confirm_shows_status_and_typing_then_drains_before_new_answer(group, error):
    h = Harness()
    if not group:
        h.chat_type, h.telegram_chat_id = "private", 7
    h.typing_release = asyncio.Event()
    body_ready = asyncio.Event()
    answer = "Подробный факт [1]. " * 400
    result = {"answer": answer, "sources": [{"file_name": "Manual.pdf", "page": 2}], "message_id": str(uuid4())}

    async def prepare(*args, **kwargs):
        assert isinstance(h.methods[0], AnswerCallbackQuery)
        assert any(item["text"] == "Готовлю ответ…" for item in h.messages.values())
        await h.typing_started.wait()
        assert h.typing is True
        body_ready.set()
        if error:
            raise httpx.ConnectError("offline failure")
        return result

    h.backend.expand_response.side_effect = prepare
    delivery = asyncio.create_task(h.click("y"))
    await asyncio.wait_for(body_ready.wait(), 2)
    await asyncio.sleep(0)
    assert not delivery.done()
    assert not any(isinstance(method, SendMessage) and method.text != "Готовлю ответ…" for method in h.methods)
    h.typing_release.set()
    await asyncio.wait_for(delivery, 2)

    assert h.typing_finished.is_set()
    assert all(task.done() and not task.cancelled() for task in h.typing_tasks)
    assert h.typing is False
    assert all(item["text"] != "Готовлю ответ…" for item in h.messages.values())
    new_answer_index = next(index for index, method in enumerate(h.methods)
                            if isinstance(method, SendMessage) and method.text != "Готовлю ответ…")
    deletion_index = next(index for index, method in enumerate(h.methods) if isinstance(method, DeleteMessage))
    assert new_answer_index < deletion_index
    h.backend.expand_response.assert_awaited_once_with(
        h.chat_id, h.message_id, owner_external_id="-100:7" if group else "7")
    delivered = list(h.messages.values())
    if error:
        assert delivered == [{"text": UNAVAILABLE, "markup": None}]
        assert not any(isinstance(method, EditMessageReplyMarkup) for method in h.methods)
    else:
        assert "".join(item["text"] for item in delivered) == answer
        assert len(delivered) > 1
        assert all(item["markup"] is None for item in delivered[:-1])
        buttons = [b for row in delivered[-1]["markup"].inline_keyboard for b in row]
        assert [b.text for b in buttons] == ["📚 Документы (1)", "👍", "👎"]
        assert not any((b.callback_data or "").startswith("exp:") for b in buttons)
    assert all(len(method.text) <= 1800 for method in h.methods if isinstance(method, SendMessage))


@pytest.mark.asyncio
@pytest.mark.parametrize("during_drain", [False, True], ids=["backend-wait", "typing-drain"])
async def test_confirm_cancellation_removes_waiting_status_and_never_delivers_result(during_drain):
    h = Harness()
    h.typing_release = asyncio.Event()
    body_ready = asyncio.Event()

    async def prepare(*args, **kwargs):
        await h.typing_started.wait()
        body_ready.set()
        if not during_drain:
            await asyncio.Event().wait()
        return h.backend.expand_response.return_value

    h.backend.expand_response.side_effect = prepare
    delivery = asyncio.create_task(h.click("y"))
    await asyncio.wait_for(body_ready.wait(), 2)
    await asyncio.sleep(0)
    delivery.cancel()
    with pytest.raises(asyncio.CancelledError):
        await delivery
    assert h.messages == {}
    assert all(task.done() for task in h.typing_tasks)
    assert all(method.text == "Готовлю ответ…" for method in h.methods if isinstance(method, SendMessage))
    assert not any(isinstance(method, EditMessageText) for method in h.methods)
    h.backend.expand_response.assert_awaited_once()
