from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.handlers.fsm import choose_topic, start_ask
from bot.states import AskFlow


@pytest.mark.asyncio
async def test_ask_flow_topic_moves_to_waiting_for_question() -> None:
    message = SimpleNamespace(answer=AsyncMock())
    state = AsyncMock()

    await start_ask(message, state)

    state.set_state.assert_awaited_with(AskFlow.waiting_for_topic)
    assert message.answer.await_count == 1

    callback = SimpleNamespace(
        data="topic:api_errors",
        message=SimpleNamespace(edit_text=AsyncMock()),
        answer=AsyncMock(),
    )

    await choose_topic(callback, state)

    state.update_data.assert_awaited_with(topic="Ошибки API")
    state.set_state.assert_awaited_with(AskFlow.waiting_for_question)
    callback.message.edit_text.assert_awaited()
    callback.answer.assert_awaited()
