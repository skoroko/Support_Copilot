from __future__ import annotations

import asyncio

import httpx
import uvicorn
from aiohttp import ThreadedResolver
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import get_settings
from bot.handlers import router
from bot.services.backend_client import BackendClient
from bot.web import build_api


async def main() -> None:
    settings = get_settings()
    telegram_session = AiohttpSession()
    # Use the Windows/system resolver. Optional aiodns installations can time out
    # even when the OS resolver and HTTPS connectivity are healthy.
    telegram_session._connector_init["resolver"] = ThreadedResolver()
    bot = Bot(token=settings.bot_token.get_secret_value(), session=telegram_session)
    http = httpx.AsyncClient(base_url=settings.backend_url, timeout=httpx.Timeout(60.0, connect=3.0))
    backend = BackendClient(settings.backend_url, client=http)
    dp = Dispatcher(storage=MemoryStorage())
    dp["backend"] = backend
    dp.include_router(router)

    api = build_api(bot, settings.internal_token.get_secret_value())
    server = uvicorn.Server(
        uvicorn.Config(api, host="0.0.0.0", port=settings.bot_api_port, log_level="info")
    )
    try:
        await asyncio.gather(dp.start_polling(bot), server.serve())
    finally:
        await http.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
