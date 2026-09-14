"""Strict, explicitly allowlisted tools for the homework 6.2 ReAct agent.

The business handlers deliberately remain in :mod:`agent_naive_tools`.  This
module only describes their stricter Chat Completions contracts, so the 6.2
increment cannot silently change the 6.1 baseline.
"""

from __future__ import annotations

from typing import Any

from app.services.agent_naive_tools import (
    get_current_time,
    search_knowledge_base,
    send_telegram_message,
)


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    """Build an OpenAI strict function schema with no undeclared arguments."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS: list[dict[str, Any]] = [
    _tool(
        "search_knowledge_base",
        "Когда нужны проверяемые сведения о MIDAS, ищет один лучший фрагмент в локальной базе знаний. "
        "На вход получает короткий текстовый запрос query и возвращает найденный ответ с источником либо честное сообщение об отсутствии надёжного совпадения. "
        "Не используй для выдуманных функций, цен, регламентов или данных, которых нет в базе.",
        {
            "query": {
                "type": "string",
                "description": "Короткий поисковый запрос по проблеме пользователя.",
            }
        },
        ["query"],
    ),
    _tool(
        "get_current_time",
        "Когда ответ зависит от текущей даты или времени, возвращает время для указанного часового пояса IANA. "
        "На вход получает timezone, например Europe/Moscow, и возвращает ISO-8601 время без сетевого запроса. "
        "Не используй как источник расписаний, сроков лицензии или иных не проверенных в базе фактов.",
        {
            "timezone": {
                "type": "string",
                "description": "Часовой пояс формата IANA, например Europe/Moscow.",
            }
        },
        ["timezone"],
    ),
    _tool(
        "send_telegram_message",
        "Когда пользователь уже явно подтвердил точные chat_id и текст, имитирует отправку сообщения в Telegram. "
        "На вход получает chat_id и text и возвращает результат безопасной локальной заглушки без обращения к Telegram API. "
        "Не вызывай без подтверждения обоих значений и не утверждай, что сообщение было доставлено реальному получателю.",
        {
            "chat_id": {"type": "string", "description": "Явно подтверждённый идентификатор чата."},
            "text": {"type": "string", "description": "Явно подтверждённый текст сообщения."},
        },
        ["chat_id", "text"],
    ),
]


# An explicit allowlist is intentionally kept next to the schemas.  No tool
# name supplied by a model is ever resolved dynamically.
DISPATCH = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "send_telegram_message": send_telegram_message,
}
