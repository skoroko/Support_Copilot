from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.enums import ParseMode

from bot.handlers._common import _render_telegram_html, format_all_sources, source_document_count, stream_to_chat


async def token_stream():
    yield "При"
    yield "вет"


@pytest.mark.asyncio
async def test_stream_to_chat_uses_draft_and_final_message() -> None:
    bot = SimpleNamespace(
        send_chat_action=AsyncMock(),
        send_message_draft=AsyncMock(),
        send_message=AsyncMock(),
    )
    message = SimpleNamespace(chat=SimpleNamespace(id=42), bot=bot)

    result = await stream_to_chat(message, token_stream())

    assert result == "Привет"
    bot.send_message_draft.assert_awaited()
    draft_ids = {call.kwargs["draft_id"] for call in bot.send_message_draft.await_args_list}
    assert len(draft_ids) == 1
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
    message.answer.assert_awaited_once_with("Готовлю ответ…")
    assert editable.edit_text.await_args_list[-1].args == ("Привет",)
    assert editable.edit_text.await_args_list[-1].kwargs == {"parse_mode": ParseMode.HTML}
    bot.send_message_draft.assert_not_awaited()
    bot.send_message.assert_not_awaited()


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

    assert editable.edit_text.await_args_list[0].args == ("**GTS",)
    assert editable.edit_text.await_args_list[0].kwargs == {}
    assert editable.edit_text.await_args_list[-1].args == ("<b>GTS NX</b> uses <code>x &lt; y</code>",)
    assert editable.edit_text.await_args_list[-1].kwargs == {"parse_mode": ParseMode.HTML}
