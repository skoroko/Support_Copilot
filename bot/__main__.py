from __future__ import annotations

import asyncio

import httpx
import redis.asyncio as redis
import uvicorn
from aiohttp import ThreadedResolver
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError

from app.observability.logging import get_logger
from bot.admission import AdmissionMiddleware
from bot.background import broadcast_worker
from bot.config import get_settings
from bot.handlers import router
from bot.services.backend_client import BackendClient
from bot.web import build_api


logger = get_logger("telegram-bot")


async def poll_with_retry(
    dispatcher: Dispatcher,
    bot: Bot,
    *,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
) -> None:
    """Keep the bot process alive across temporary Telegram network failures."""
    delay = initial_delay
    while True:
        try:
            await dispatcher.start_polling(bot)
            return
        except asyncio.CancelledError:
            raise
        except TelegramNetworkError as exc:
            logger.warning(
                "telegram_polling_unavailable",
                error_type=type(exc).__name__,
                retry_in_seconds=delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)


async def main() -> None:
    settings = get_settings()
    telegram_session = AiohttpSession()
    # Use the Windows/system resolver. Optional aiodns installations can time out
    # even when the OS resolver and HTTPS connectivity are healthy.
    telegram_session._connector_init["resolver"] = ThreadedResolver()
    bot = Bot(token=settings.bot_token.get_secret_value(), session=telegram_session)
    http = httpx.AsyncClient(base_url=settings.backend_url, timeout=httpx.Timeout(60.0, connect=3.0))
    backend = BackendClient(
        settings.backend_url,
        client=http,
        internal_token=settings.internal_token.get_secret_value(),
        admin_token=settings.admin_token.get_secret_value(),
    )
    dp = Dispatcher()
    dp.update.outer_middleware(
        AdmissionMiddleware(
            user_ids=settings.bot_allowed_user_ids,
            chat_ids=settings.bot_allowed_chat_ids,
        )
    )
    dp["backend"] = backend
    dp["bot_admin_ids"] = settings.bot_admin_ids
    dp.include_router(router)

    idempotency_store = redis.from_url(settings.redis_url, decode_responses=True)
    api = build_api(
        bot,
        settings.internal_token.get_secret_value(),
        idempotency_store=idempotency_store,
    )
    server = uvicorn.Server(
        uvicorn.Config(api, host="0.0.0.0", port=settings.bot_api_port, log_level="info")
    )
    try:
        await asyncio.gather(
            poll_with_retry(dp, bot),
            server.serve(),
            broadcast_worker(bot, backend, settings.broadcast_poll_seconds),
        )
    finally:
        await idempotency_store.aclose()
        await http.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
