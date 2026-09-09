from __future__ import annotations

import asyncio

import httpx
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

from app.observability.logging import get_logger
from bot.services.backend_client import BackendClient


logger = get_logger("bot-broadcast")


async def broadcast_worker(bot: Bot, backend: BackendClient, poll_seconds: float) -> None:
    while True:
        try:
            items = await backend.get_pending_broadcasts(limit=10)
            for item in items:
                broadcast_id = str(item["id"])
                await backend.update_broadcast_status(broadcast_id, "processing")
                users = await backend.get_admin_users(limit=1000, interface=str(item["interface"]))
                failed = 0
                for user in users:
                    try:
                        await bot.send_message(chat_id=int(user["owner_external_id"]), text=str(item["message"]))
                    except (TelegramAPIError, TypeError, ValueError):
                        failed += 1
                await backend.update_broadcast_status(broadcast_id, "failed" if failed else "sent")
                logger.info(
                    "broadcast_completed",
                    broadcast_id=broadcast_id,
                    recipients=len(users),
                    failed=failed,
                )
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:
            logger.warning("broadcast_poll_failed", error_type=type(exc).__name__)
        await asyncio.sleep(poll_seconds)
