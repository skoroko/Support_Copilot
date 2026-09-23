from __future__ import annotations

import asyncio
from collections.abc import Iterable

import httpx
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

from app.chat.domain import telegram_chat_id_from_owner
from app.observability.logging import get_logger
from bot.services.backend_client import BackendClient


logger = get_logger("bot-broadcast")


def broadcast_chat_ids(users: Iterable[dict]) -> tuple[list[int], int]:
    """Telegram chat ids that each receive one broadcast copy.

    Group members have ``"<chat_id>:<sender_id>"`` owner keys that all resolve
    to the same group chat, so targets are deduplicated per Telegram chat.
    Returns ``(chat_ids, skipped_rows)``.
    """

    targets: list[int] = []
    seen: set[int] = set()
    skipped = 0
    for user in users:
        try:
            target = telegram_chat_id_from_owner(str(user.get("owner_external_id") or ""))
        except (AttributeError, TypeError, ValueError):
            skipped += 1
            continue
        if target in seen:
            continue
        seen.add(target)
        targets.append(target)
    return targets, skipped


async def broadcast_worker(bot: Bot, backend: BackendClient, poll_seconds: float) -> None:
    while True:
        try:
            items = await backend.get_pending_broadcasts(limit=10)
            for item in items:
                broadcast_id = str(item["id"])
                await backend.update_broadcast_status(broadcast_id, "processing")
                users = await backend.get_admin_users(limit=1000, interface=str(item["interface"]))
                targets, failed = broadcast_chat_ids(users)
                for target in targets:
                    try:
                        await bot.send_message(chat_id=target, text=str(item["message"]))
                    except TelegramAPIError:
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
