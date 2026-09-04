from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.handlers._common import stream_to_chat


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
    bot.send_message.assert_awaited_once_with(chat_id=42, text="Привет")
