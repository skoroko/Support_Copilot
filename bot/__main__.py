from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import get_settings
from bot.handlers import router
from bot.services.backend_client import BackendClient


async def main() -> None:
    settings = get_settings()
    bot = Bot(token=settings.bot_token.get_secret_value())
    backend = BackendClient(settings.backend_url)
    dp = Dispatcher(storage=MemoryStorage())
    dp["backend"] = backend
    dp.include_router(router)

    try:
        await dp.start_polling(bot)
    finally:
        await backend.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
