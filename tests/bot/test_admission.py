from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Update

from bot.admission import AdmissionMiddleware


def update(*, user_id: int | None, chat_id: int | None, chat_type: str = "private", bot: bool = False):
    user = None if user_id is None else SimpleNamespace(id=user_id, is_bot=bot)
    chat = None if chat_id is None else SimpleNamespace(id=chat_id, type=chat_type)
    return SimpleNamespace(event=SimpleNamespace(from_user=user, chat=chat))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "admitted"),
    [
        (update(user_id=10, chat_id=10), True),
        (update(user_id=11, chat_id=11), False),
        (update(user_id=10, chat_id=11), False),
        (update(user_id=10, chat_id=-100, chat_type="supergroup"), True),
        (update(user_id=10, chat_id=-101, chat_type="group"), False),
        (update(user_id=11, chat_id=-100, chat_type="group"), False),
        (update(user_id=None, chat_id=-100, chat_type="group"), False),
        (update(user_id=10, chat_id=-100, chat_type="channel"), False),
        (update(user_id=10, chat_id=10, bot=True), False),
        (update(user_id=10, chat_id=None), False),
    ],
)
async def test_update_admission(event, admitted: bool) -> None:
    handler = AsyncMock(return_value="handled")
    middleware = AdmissionMiddleware(user_ids=[10], chat_ids=[-100])

    result = await middleware(handler, event, {})

    assert (result == "handled") is admitted
    assert handler.await_count == int(admitted)


@pytest.mark.asyncio
async def test_callback_uses_clicking_user_and_message_chat() -> None:
    handler = AsyncMock(return_value="handled")
    middleware = AdmissionMiddleware(user_ids=[10], chat_ids=[-100])
    callback = SimpleNamespace(
        from_user=SimpleNamespace(id=10, is_bot=False),
        message=SimpleNamespace(chat=SimpleNamespace(id=-100, type="supergroup")),
    )

    assert await middleware(handler, SimpleNamespace(event=callback), {}) == "handled"
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_configured_users_denies_every_update() -> None:
    handler = AsyncMock()
    middleware = AdmissionMiddleware(user_ids=[], chat_ids=[])
    await middleware(handler, update(user_id=10, chat_id=10), {})
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_update_type_is_denied() -> None:
    handler = AsyncMock()
    middleware = AdmissionMiddleware(user_ids=[10], chat_ids=[])
    assert await middleware(handler, Update(update_id=1), {}) is None
    handler.assert_not_awaited()
