from __future__ import annotations

import asyncio
from collections.abc import Iterable
from uuid import uuid4

import httpx
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest, TelegramForbiddenError

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
            items = await backend.claim_broadcasts(limit=10)
            for item in items:
                await deliver_broadcast(bot, backend, item)
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:
            logger.warning("broadcast_poll_failed", error_type=type(exc).__name__)
        except Exception as exc:  # noqa: BLE001 - keep polling after an unexpected backend/bot failure
            logger.warning("broadcast_poll_failed", error_type=type(exc).__name__)
        await asyncio.sleep(poll_seconds)


async def deliver_broadcast(bot: Bot, backend: BackendClient, item: dict) -> dict:
    """Resume only recipients with a durable pending state under this lease."""

    broadcast_id = str(item["id"])
    lease_id = str(item["lease_id"])
    for recipient in item["recipients"]:
        if recipient["status"] != "pending":
            continue
        chat_id = int(recipient["chat_id"])
        attempt_id = str(uuid4())
        # Persist intent before Telegram: if either response is lost, an
        # expired attempt becomes unknown and will never be sent twice.
        await backend.update_broadcast_recipient(
            broadcast_id, lease_id, chat_id, attempt_id, "in_flight",
        )
        try:
            await bot.send_message(chat_id=chat_id, text=str(item["message"]))
        except (TelegramBadRequest, TelegramForbiddenError):
            status = "failed"
        except TelegramAPIError:
            # A network failure may occur after Telegram accepted the message.
            status = "unknown"
        except Exception as exc:  # noqa: BLE001 - delivery may already have happened
            logger.warning("broadcast_delivery_uncertain", error_type=type(exc).__name__)
            status = "unknown"
        else:
            status = "sent"
        await backend.update_broadcast_recipient(
            broadcast_id, lease_id, chat_id, attempt_id, status,
        )
    completed = await backend.finish_broadcast(broadcast_id, lease_id)
    logger.info(
        "broadcast_completed",
        broadcast_id=broadcast_id,
        recipients=len(item["recipients"]),
        status=completed["status"],
    )
    return completed
