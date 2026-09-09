from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from aiogram.exceptions import TelegramNetworkError

from bot.__main__ import poll_with_retry


@pytest.mark.asyncio
async def test_poll_with_retry_recovers_from_temporary_network_failure() -> None:
    dispatcher = AsyncMock()
    dispatcher.start_polling.side_effect = [
        TelegramNetworkError(method="getMe", message="temporarily unavailable"),
        None,
    ]

    with patch("bot.__main__.asyncio.sleep", new=AsyncMock()) as sleep:
        await poll_with_retry(dispatcher, AsyncMock(), initial_delay=1.0, max_delay=30.0)

    assert dispatcher.start_polling.await_count == 2
    sleep.assert_awaited_once_with(1.0)
