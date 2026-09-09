from __future__ import annotations

from html import escape
from typing import Any

import httpx
from aiogram import Router
from aiogram.enums import ParseMode
from aiogram.filters import BaseFilter, Command, CommandObject
from aiogram.types import Message

from bot.services.backend_client import BackendClient


class IsAdmin(BaseFilter):
    def __init__(self, admin_ids: list[int] | None = None) -> None:
        self.admin_ids = frozenset(admin_ids) if admin_ids is not None else None

    async def __call__(self, message: Message, bot_admin_ids: list[int] | None = None) -> bool:
        allowed = self.admin_ids if self.admin_ids is not None else frozenset(bot_admin_ids or [])
        return message.from_user is not None and message.from_user.id in allowed


router = Router(name="admin")
router.message.filter(IsAdmin())


def _admin_error(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"Backend отклонил admin-запрос (HTTP {exc.response.status_code})."
    return "Admin API временно недоступен. Попробуйте позже."


@router.message(Command("stats"))
async def stats(message: Message, backend: BackendClient) -> None:
    try:
        data = await backend.get_admin_stats()
    except httpx.HTTPError as exc:
        await message.answer(_admin_error(exc))
        return
    text = (
        "<b>Статистика за 24 часа</b>\n"
        f"Сообщений: <code>{int(data.get('total_messages', 0))}</code>\n"
        f"Активных пользователей: <code>{int(data.get('active_users', 0))}</code>\n"
        f"Средняя задержка: <code>{float(data.get('avg_latency_ms', 0)):.2f} мс</code>\n"
        f"Блокировки модерации: <code>{float(data.get('moderation_block_rate', 0)):.1%}</code>\n"
        f"Положительный feedback: <code>{float(data.get('feedback_up_ratio', 0)):.1%}</code>"
    )
    await message.answer(text, parse_mode=ParseMode.HTML)


@router.message(Command("users"))
async def users(message: Message, backend: BackendClient) -> None:
    try:
        rows = (await backend.get_admin_users(limit=50))[:10]
    except httpx.HTTPError as exc:
        await message.answer(_admin_error(exc))
        return
    if not rows:
        await message.answer("Пользователей пока нет.")
        return
    lines = ["ID                 Чаты  Последняя активность"]
    for row in rows:
        owner = str(row.get("owner_external_id", ""))[:18]
        count = int(row.get("chat_count", 0))
        seen = str(row.get("last_seen_at", ""))[:19].replace("T", " ")
        lines.append(f"{owner:<18} {count:>4}  {seen}")
    await message.answer(f"<b>Последние пользователи</b>\n<pre>{escape(chr(10).join(lines))}</pre>", parse_mode=ParseMode.HTML)


@router.message(Command("broadcast"))
async def broadcast(message: Message, command: CommandObject, backend: BackendClient) -> None:
    text = (command.args or "").strip()
    if not text:
        await message.answer("Использование: /broadcast &lt;текст&gt;", parse_mode=ParseMode.HTML)
        return
    try:
        item: dict[str, Any] = await backend.create_broadcast(text, interface="telegram")
    except httpx.HTTPError as exc:
        await message.answer(_admin_error(exc))
        return
    await message.answer(f"Рассылка поставлена в очередь: <code>{escape(str(item.get('id', '')))}</code>", parse_mode=ParseMode.HTML)
