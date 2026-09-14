from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from support_copilot.tools.handlers import search_knowledge_base as _search_local


def search_knowledge_base(query: str) -> str:
    """Return the best matching fragment from the existing local support KB."""
    payload = _search_local(query=query)
    if not payload["results"] or payload["results"][0]["score"] < 4:
        return "В базе знаний нет надежного совпадения; нужна проверка инженером поддержки."
    item = payload["results"][0]
    return f'{item["title"]}: {item["answer"]} Источник: {item["source"]}.'


def get_current_time(timezone: str = "Europe/Moscow") -> str:
    return datetime.now(ZoneInfo(timezone)).isoformat()


def send_telegram_message(chat_id: str, text: str) -> str:
    print(f"[TELEGRAM → {chat_id}] {text}")
    return f"Сообщение отправлено в {chat_id}"


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}}


TOOLS = [
    _tool("search_knowledge_base", "Ищет ответ в локальной базе знаний Support Copilot и возвращает один лучший фрагмент. Вызывай, когда для ответа нужны проверяемые сведения о поддержке MIDAS.", {"query": {"type": "string", "description": "Краткий поисковый запрос"}}, ["query"]),
    _tool("get_current_time", "Возвращает текущее время в указанном часовом поясе IANA без сетевого запроса. Вызывай, когда задача зависит от текущей даты или времени.", {"timezone": {"type": "string", "description": "Часовой пояс IANA, например Europe/Moscow", "default": "Europe/Moscow"}}, []),
    _tool("send_telegram_message", "Имитирует отправку сообщения в Telegram и не обращается к Telegram API. Вызывай только после явного подтверждения пользователем текста и chat_id.", {"chat_id": {"type": "string", "description": "Идентификатор чата"}, "text": {"type": "string", "description": "Подтверждённый текст сообщения"}}, ["chat_id", "text"]),
]
DISPATCH = {"search_knowledge_base": search_knowledge_base, "get_current_time": get_current_time,
            "send_telegram_message": send_telegram_message}
