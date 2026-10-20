from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import DeleteMessage, EditMessageText, SendMessage

from bot.handlers._common import (
    _render_telegram_html,
    format_all_sources,
    send_rendered_answer,
    source_document_count,
    stream_to_chat,
)
from bot.keyboards.inline import answer_kb, approval_kb
from bot.services.backend_client import StreamReplace


async def token_stream():
    yield "При"
    yield "вет"


@pytest.mark.asyncio
async def test_transport_without_editable_placeholder_sends_one_final_message() -> None:
    bot = SimpleNamespace(
        send_chat_action=AsyncMock(),
        send_message_draft=AsyncMock(),
        send_message=AsyncMock(),
    )
    message = SimpleNamespace(chat=SimpleNamespace(id=42), bot=bot)

    result = await stream_to_chat(message, token_stream())

    assert result == "Привет"
    bot.send_message_draft.assert_not_awaited()
    bot.send_message.assert_awaited_once_with(
        chat_id=42,
        text="Привет",
        parse_mode=ParseMode.HTML,
    )


@pytest.mark.asyncio
async def test_stream_to_chat_appends_sources_shown_to_user() -> None:
    editable = SimpleNamespace(edit_text=AsyncMock())
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message_draft=AsyncMock())
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        bot=bot,
        answer=AsyncMock(return_value=editable),
    )

    async def cited_stream():
        yield "Привет [1]"

    result = await stream_to_chat(
        message,
        cited_stream(),
        sources_factory=lambda: [{"file_name": "manual.pdf", "page": 7}],
    )

    assert result == "Привет [1]"
    assert editable.edit_text.await_args_list[-1].args[0].endswith(
        "Источники:\n[1] manual.pdf, PDF стр. 7"
    )
    assert editable.edit_text.await_args_list[-1].kwargs["parse_mode"] == ParseMode.HTML


@pytest.mark.asyncio
async def test_stream_to_chat_shows_only_sources_cited_with_original_numbers() -> None:
    editable = SimpleNamespace(edit_text=AsyncMock())
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message_draft=AsyncMock())
    message = SimpleNamespace(chat=SimpleNamespace(id=42), bot=bot, answer=AsyncMock(return_value=editable))
    sources = [{"file_name": f"manual-{index}.pdf", "page": index} for index in range(1, 11)]

    async def cited_stream():
        yield "Ответ по [2] и [7]."

    await stream_to_chat(message, cited_stream(), sources_factory=lambda: sources)

    final_text = editable.edit_text.await_args_list[-1].args[0]
    assert "Источники:\n[2] manual-2.pdf, PDF стр. 2\n[7] manual-7.pdf, PDF стр. 7" in final_text
    assert "[1] manual-1.pdf" not in final_text
    assert "[10] manual-10.pdf" not in final_text


@pytest.mark.asyncio
async def test_documents_button_replaces_duplicate_bibliography_in_answer() -> None:
    from uuid import uuid4

    editable = SimpleNamespace(edit_text=AsyncMock())
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
        answer=AsyncMock(return_value=editable),
    )
    sources = [{"file_name": "Manual.pdf", "page": 7}]
    markup = answer_kb(uuid4(), uuid4(), 1, expandable=True)

    async def cited_stream():
        yield "Подтверждённый факт [1]."

    result = await stream_to_chat(
        message, cited_stream(), sources_factory=lambda: sources,
        reply_markup_factory=lambda: markup,
    )

    final = editable.edit_text.await_args_list[-1]
    assert result == "Подтверждённый факт [1]."
    assert final.args[0] == result
    assert final.kwargs["reply_markup"] == markup
    assert "Документы (1)" in markup.inline_keyboard[0][0].text
    assert format_all_sources(sources) == "Документы:\n[1] Manual.pdf, PDF стр. 7"


def test_format_all_sources_keeps_full_original_numbering() -> None:
    sources = [{"file_name": f"manual-{index}.pdf", "page": index} for index in range(1, 4)]

    assert format_all_sources(sources) == (
        "Документы:\n[1] manual-1.pdf, PDF стр. 1\n[2] manual-2.pdf, PDF стр. 2\n"
        "[3] manual-3.pdf, PDF стр. 3"
    )


def test_format_all_sources_groups_same_document_and_keeps_citation_numbers() -> None:
    sources = [
        {"file_name": "GTS NX manual.pdf", "page": 3},
        {"file_name": "other.pdf", "page": 9},
        {"file_name": "GTS NX manual.pdf", "page": 7},
    ]

    assert format_all_sources(sources) == (
        "Документы:\n[1, 3] GTS NX manual.pdf, PDF стр. 3; PDF стр. 7\n[2] other.pdf, PDF стр. 9"
    )
    assert source_document_count(sources) == 2


def test_format_all_sources_distinguishes_pdf_and_printed_pages() -> None:
    sources = [
        {
            "file_name": "GTS NX manual.pdf",
            "page": 32,
            "printed_page": 45,
        }
    ]

    assert format_all_sources(sources) == (
        "Документы:\n[1] GTS NX manual.pdf, PDF стр. 32 (в документе 45)"
    )


def test_format_all_sources_keeps_physical_and_printed_page_pairs() -> None:
    sources = [
        {"file_name": "GTS NX manual.pdf", "page": 32, "printed_page": 45},
        {"file_name": "GTS NX manual.pdf", "page": 60, "printed_page": 79},
    ]

    assert format_all_sources(sources) == (
        "Документы:\n[1, 2] GTS NX manual.pdf, PDF стр. 32 (в документе 45); "
        "PDF стр. 60 (в документе 79)"
    )


@pytest.mark.asyncio
async def test_stream_to_chat_replace_event_rewrites_draft() -> None:
    editable = SimpleNamespace(edit_text=AsyncMock())
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message_draft=AsyncMock(), send_message=AsyncMock())
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        bot=bot,
        answer=AsyncMock(return_value=editable),
    )

    async def replacing_stream():
        yield "Чер"
        yield "новик"
        yield StreamReplace("Финальный ответ")

    result = await stream_to_chat(message, replacing_stream())

    assert result == "Финальный ответ"
    assert editable.edit_text.await_args_list[-1].args == ("Финальный ответ",)
    assert editable.edit_text.await_args_list[-1].kwargs == {"parse_mode": ParseMode.HTML}
    bot.send_message.assert_not_awaited()
    bot.send_chat_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_to_chat_edits_one_response_message_when_available() -> None:
    editable = SimpleNamespace(edit_text=AsyncMock())
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message_draft=AsyncMock(), send_message=AsyncMock())
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        bot=bot,
        answer=AsyncMock(return_value=editable),
    )

    result = await stream_to_chat(message, token_stream())

    assert result == "Привет"
    assert [call.args[0] for call in message.answer.await_args_list] == ["Готовлю ответ…", "При"]
    assert editable.edit_text.await_args_list[-1].args == ("Привет",)
    assert editable.edit_text.await_args_list[-1].kwargs == {"parse_mode": ParseMode.HTML}
    bot.send_message_draft.assert_not_awaited()
    bot.send_message.assert_not_awaited()
    bot.send_chat_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_to_chat_appends_notice_to_final_message() -> None:
    editable = SimpleNamespace(edit_text=AsyncMock())
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message_draft=AsyncMock())
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        bot=bot,
        answer=AsyncMock(return_value=editable),
    )

    result = await stream_to_chat(
        message,
        token_stream(),
        notice_factory=lambda: "База знаний временно недоступна",
    )

    assert result == "Привет"
    final_text = editable.edit_text.await_args_list[-1].args[0]
    assert final_text.endswith("\n\nБаза знаний временно недоступна")


@pytest.mark.asyncio
async def test_stream_to_chat_skips_empty_notice() -> None:
    editable = SimpleNamespace(edit_text=AsyncMock())
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message_draft=AsyncMock())
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        bot=bot,
        answer=AsyncMock(return_value=editable),
    )

    await stream_to_chat(message, token_stream(), notice_factory=lambda: "")

    assert editable.edit_text.await_args_list[-1].args[0] == "Привет"


def test_render_telegram_html_supports_bold_inline_code_and_escapes_content() -> None:
    assert _render_telegram_html("**GTS & NX**: `x < 1` <script>") == (
        "<b>GTS &amp; NX</b>: <code>x &lt; 1</code> &lt;script&gt;"
    )


@pytest.mark.asyncio
async def test_stream_to_chat_keeps_partial_draft_plain_and_formats_final_message() -> None:
    editable = SimpleNamespace(edit_text=AsyncMock())
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message_draft=AsyncMock(), send_message=AsyncMock())
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        bot=bot,
        answer=AsyncMock(return_value=editable),
    )

    async def markdown_stream():
        yield "**GTS"
        yield " NX** uses `x < y`"

    await stream_to_chat(message, markdown_stream())

    assert message.answer.await_args_list[1].args == ("**GTS",)
    assert message.answer.await_args_list[1].kwargs == {"parse_mode": None}
    assert editable.edit_text.await_args_list[-1].args == ("<b>GTS NX</b> uses <code>x &lt; y</code>",)
    assert editable.edit_text.await_args_list[-1].kwargs == {"parse_mode": ParseMode.HTML}


@pytest.mark.asyncio
async def test_chunked_approval_preserves_literal_text_and_last_chunk_buttons():
    from html import unescape
    from uuid import uuid4

    markup = approval_kb(uuid4(), uuid4())
    message = SimpleNamespace(chat=SimpleNamespace(id=42), answer=AsyncMock())
    text = "**E50ref** — `literal < 0.5`.\n\n" * 150
    await send_rendered_answer(message, text, reply_markup=markup)
    calls = message.answer.await_args_list
    assert len(calls) > 1
    assert "".join(unescape(call.args[0]) for call in calls) == text
    assert all(call.kwargs["reply_markup"] is None for call in calls[:-1])
    assert calls[-1].kwargs["reply_markup"] is markup


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["Ответ.", StreamReplace("Ответ."), StreamReplace("")],
                         ids=["token", "replace", "empty-replace"])
async def test_fallback_joins_inflight_typing_before_draft_or_final_delivery(body):
    started = asyncio.Event()
    stopped = asyncio.Event()
    release = asyncio.Event()
    typing_tasks = []

    async def typing(**_kwargs):
        typing_tasks.append(asyncio.current_task())
        started.set()
        try:
            await release.wait()
        finally:
            stopped.set()

    async def delivered(**_kwargs):
        assert stopped.is_set()
        assert typing_tasks[0].done()

    bot = SimpleNamespace(send_chat_action=AsyncMock(side_effect=typing),
                          send_message_draft=AsyncMock(side_effect=delivered),
                          send_message=AsyncMock(side_effect=delivered))
    message = SimpleNamespace(chat=SimpleNamespace(id=42), bot=bot)

    async def tokens():
        await started.wait()
        asyncio.get_running_loop().call_soon(release.set)
        yield body

    await asyncio.wait_for(stream_to_chat(message, tokens()), timeout=2)
    bot.send_message.assert_awaited_once()
    bot.send_chat_action.assert_awaited_once()
    assert not typing_tasks[0].cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["backend-error", "caller-cancelled"])
async def test_fallback_typing_is_joined_on_stream_failure_or_cancellation(cancel):
    started = asyncio.Event()
    stopped = asyncio.Event()
    typing_tasks = []

    async def typing(**_kwargs):
        typing_tasks.append(asyncio.current_task())
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    bot = SimpleNamespace(send_chat_action=AsyncMock(side_effect=typing),
                          send_message_draft=AsyncMock(), send_message=AsyncMock())
    message = SimpleNamespace(chat=SimpleNamespace(id=42), bot=bot)

    async def tokens():
        await started.wait()
        if cancel:
            await asyncio.Event().wait()
        raise httpx.ReadError("offline stream failure")
        yield ""  # Keep the failing provider double an async iterator.

    delivery = asyncio.create_task(stream_to_chat(message, tokens()))
    await asyncio.wait_for(started.wait(), timeout=2)
    if cancel:
        delivery.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else httpx.ReadError):
        await asyncio.wait_for(delivery, timeout=2)
    assert stopped.is_set()
    assert typing_tasks[0].cancelled()
    bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_fallback_stops_waiting_refresher_before_replacement_delivery():
    started = asyncio.Event()
    typing_tasks = []

    async def typing(**_kwargs):
        typing_tasks.append(asyncio.current_task())
        started.set()

    async def delivered(**_kwargs):
        assert typing_tasks[0].done()

    bot = SimpleNamespace(send_chat_action=AsyncMock(side_effect=typing),
                          send_message_draft=AsyncMock(side_effect=delivered),
                          send_message=AsyncMock(side_effect=delivered))
    message = SimpleNamespace(chat=SimpleNamespace(id=42), bot=bot)

    async def tokens():
        await started.wait()
        yield StreamReplace("Ответ.")

    assert await asyncio.wait_for(stream_to_chat(message, tokens()), timeout=2) == "Ответ."
    bot.send_chat_action.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["replacement", "backend-error"])
async def test_editable_placeholder_and_typing_are_shown_during_slow_backend(fails):
    editable = SimpleNamespace(edit_text=AsyncMock())
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message_draft=AsyncMock())
    message = SimpleNamespace(chat=SimpleNamespace(id=42), bot=bot,
                              answer=AsyncMock(return_value=editable))

    async def tokens():
        # Let the typing refresher run while the backend waits.
        await asyncio.sleep(0)
        if fails:
            raise httpx.ReadError("offline stream failure")
        yield StreamReplace("Готовый ответ.")

    if fails:
        with pytest.raises(httpx.ReadError):
            await stream_to_chat(message, tokens())
    else:
        assert await stream_to_chat(message, tokens()) == "Готовый ответ."
        assert editable.edit_text.await_args_list[-1].args[0] == "Готовый ответ."
    assert message.answer.await_args_list[0].args[0] == "Готовлю ответ…"
    assert message.answer.await_count == (1 if fails else 2)
    bot.send_chat_action.assert_awaited_once()
    bot.send_message_draft.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["Ответ.", StreamReplace("Ответ."), StreamReplace("Ответ. " * 700),
                                  StreamReplace("")], ids=["token", "replace", "long", "empty"])
@pytest.mark.parametrize("group", [False, True], ids=["private", "group"])
async def test_new_answer_clears_drained_typing_then_removes_placeholder(body, group):
    started, release = asyncio.Event(), asyncio.Event()
    events = []
    responses = []
    markup = SimpleNamespace(inline_keyboard=[])
    placeholder = SimpleNamespace(edit_text=AsyncMock(side_effect=AssertionError("Never edit waiting status")))

    async def delete_placeholder():
        events.append("delete-placeholder")

    placeholder.delete = AsyncMock(side_effect=delete_placeholder)

    async def typing(**_kwargs):
        events.append("typing-request")
        started.set()
        await release.wait()
        events.append("typing-complete")

    async def send(text, **kwargs):
        if text == "Готовлю ответ…":
            events.append("placeholder")
            return placeholder
        assert events.index("typing-complete") > events.index("typing-request")
        events.append("new-answer")
        assert len(text) <= 1800
        item = SimpleNamespace(text=text, markup=kwargs.get("reply_markup"))

        async def edit(text, **kwargs):
            assert "delete-placeholder" in events
            item.text, item.markup = text, kwargs.get("reply_markup")

        item.edit_text = AsyncMock(side_effect=edit)
        responses.append(item)
        return item

    bot = SimpleNamespace(send_chat_action=AsyncMock(side_effect=typing), send_message_draft=AsyncMock())
    message = SimpleNamespace(chat=SimpleNamespace(id=42, type="supergroup" if group else "private"), bot=bot,
                              answer=AsyncMock(side_effect=send), reply=AsyncMock(side_effect=send))

    async def tokens():
        await started.wait()
        asyncio.get_running_loop().call_soon(release.set)
        yield body

    result = await asyncio.wait_for(stream_to_chat(message, tokens(), reply_markup_factory=lambda: markup), 2)
    expected = (body.text if isinstance(body, StreamReplace) else body) or "Backend вернул пустой ответ."
    assert "".join(item.text for item in responses) == expected
    assert result == (body.text if isinstance(body, StreamReplace) else body)
    assert responses[-1].markup is markup
    assert all(item.markup is None for item in responses[:-1])
    assert events.index("new-answer") < events.index("delete-placeholder")
    placeholder.delete.assert_awaited_once()
    placeholder.edit_text.assert_not_awaited()
    bot.send_chat_action.assert_awaited_once()
    bot.send_message_draft.assert_not_awaited()
    (message.answer if group else message.reply).assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["send", "delete"])
async def test_transition_failure_never_deletes_status_before_answer_or_duplicates_answer(failure):
    started = asyncio.Event()
    placeholder = SimpleNamespace(delete=AsyncMock())
    answer = SimpleNamespace(edit_text=AsyncMock())
    send_error = TelegramBadRequest(method=SendMessage(chat_id=42, text="Ответ."), message="offline failure")
    if failure == "delete":
        placeholder.delete.side_effect = TelegramBadRequest(
            method=DeleteMessage(chat_id=42, message_id=1), message="offline failure")
    message = SimpleNamespace(chat=SimpleNamespace(id=42),
                              answer=AsyncMock(side_effect=[placeholder, send_error if failure == "send" else answer]),
                              bot=SimpleNamespace(send_chat_action=AsyncMock(side_effect=lambda **_: started.set())))

    async def tokens():
        await started.wait()
        yield StreamReplace("Ответ.")

    if failure == "send":
        with pytest.raises(TelegramBadRequest):
            await stream_to_chat(message, tokens())
        placeholder.delete.assert_not_awaited()
    else:
        assert await stream_to_chat(message, tokens()) == "Ответ."
        placeholder.delete.assert_awaited_once()
        answer.edit_text.assert_awaited_once()
    assert message.answer.await_count == 2


@pytest.mark.asyncio
async def test_cancelling_while_draining_action_never_publishes_body():
    started, body_ready, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    placeholder = SimpleNamespace(delete=AsyncMock(), edit_text=AsyncMock())

    async def typing(**_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    message = SimpleNamespace(chat=SimpleNamespace(id=42), answer=AsyncMock(return_value=placeholder),
                              bot=SimpleNamespace(send_chat_action=AsyncMock(side_effect=typing)))

    async def tokens():
        await started.wait()
        body_ready.set()
        yield StreamReplace("Ответ.")

    delivery = asyncio.create_task(stream_to_chat(message, tokens()))
    await asyncio.wait_for(body_ready.wait(), 2)
    await asyncio.sleep(0)
    delivery.cancel()
    with pytest.raises(asyncio.CancelledError):
        await delivery
    assert stopped.is_set()
    message.answer.assert_awaited_once_with("Готовлю ответ…")
    placeholder.delete.assert_not_awaited()
    placeholder.edit_text.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["short", "repeat", "long"])
async def test_unchanged_answer_edits_never_send_duplicate_content(case, monkeypatch):
    import bot.handlers._common as common

    monkeypatch.setattr(common, "EDIT_DEBOUNCE_SECONDS", 0)
    placeholder = SimpleNamespace(delete=AsyncMock())
    visible = []
    body = "Ответ. " * (700 if case == "long" else 1)

    async def send(text, **kwargs):
        if text == "Готовлю ответ…":
            return placeholder
        item = SimpleNamespace(text=text, markup=kwargs.get("reply_markup"))

        async def edit(text, **kwargs):
            if text == item.text and kwargs.get("reply_markup") == item.markup:
                raise TelegramBadRequest(
                    method=EditMessageText(chat_id=42, message_id=1, text=text),
                    message="Bad Request: message is not modified",
                )
            item.text, item.markup = text, kwargs.get("reply_markup")

        item.edit_text = AsyncMock(side_effect=edit)
        visible.append(item)
        return item

    message = SimpleNamespace(chat=SimpleNamespace(id=42), answer=AsyncMock(side_effect=send),
                              bot=SimpleNamespace(send_chat_action=AsyncMock()))

    async def tokens():
        yield StreamReplace(body)
        if case == "repeat":
            yield StreamReplace(body)

    assert await stream_to_chat(message, tokens()) == body
    assert "".join(item.text for item in visible) == body
    if case != "long":
        assert len(visible) == 1
    placeholder.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_waiting_status_rotates_while_backend_stream_is_silent(monkeypatch):
    monkeypatch.setattr("bot.handlers._common.STATUS_ROTATE_SECONDS", 0.01)
    formulated = asyncio.Event()

    async def status_edited(text, **_kwargs):
        if text == "Формулирую ответ…":
            formulated.set()

    placeholder = SimpleNamespace(edit_text=AsyncMock(side_effect=status_edited), delete=AsyncMock())
    answer = SimpleNamespace(edit_text=AsyncMock())
    sends: list[str] = []

    async def send(text, **_kwargs):
        sends.append(text)
        return placeholder if text == "Готовлю ответ…" else answer

    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
        answer=AsyncMock(side_effect=send),
    )

    async def tokens():
        # Wait for the state we assert, rather than racing three timer ticks
        # against a 60ms sleep on a loaded/low-resolution event loop.
        await asyncio.wait_for(formulated.wait(), timeout=2)
        yield "Ответ."

    assert await stream_to_chat(message, tokens()) == "Ответ."

    statuses = [call.args[0] for call in placeholder.edit_text.await_args_list]
    assert statuses
    assert statuses[0] == "Ищу в мануалах…"
    assert statuses[-1] == "Формулирую ответ…"
    assert answer.edit_text.await_args_list[-1].args[0] == "Ответ."


@pytest.mark.asyncio
async def test_waiting_status_stops_on_backend_failure(monkeypatch):
    monkeypatch.setattr("bot.handlers._common.STATUS_ROTATE_SECONDS", 0.01)
    placeholder = SimpleNamespace(edit_text=AsyncMock(), delete=AsyncMock())
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
        answer=AsyncMock(return_value=placeholder),
    )

    async def tokens():
        await asyncio.sleep(0.05)
        raise httpx.ReadError("offline stream failure")
        yield ""  # Keep the failing provider double an async iterator.

    with pytest.raises(httpx.ReadError):
        await stream_to_chat(message, tokens())

    # No status edit may land after the turn failed and the rotation stopped.
    await asyncio.sleep(0.05)
    edits = placeholder.edit_text.await_count
    await asyncio.sleep(0.05)
    assert placeholder.edit_text.await_count == edits
