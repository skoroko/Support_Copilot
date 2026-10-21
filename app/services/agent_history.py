"""Project durable dialogue into a fresh, bounded model conversation."""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

PARAMETER_CLARIFICATION = (
    "Уточните, пожалуйста, о каком продукте MIDAS и какой задаче или операции идёт речь. "
    "Какие параметры вы хотите задать? Если есть ошибка или название окна настроек, укажите их."
)


def needs_parameter_clarification(messages: list[AnyMessage]) -> bool:
    """Recognize an object-free first question without guessing its subject.

    Earlier user context and media may supply the missing object. Specific
    questions fall through to normal retrieval, even without a product name.
    """
    humans = [message for message in messages if message.type == "human"]
    if len(humans) != 1 or not isinstance(humans[0].content, str):
        return False
    text = " ".join(humans[0].content.casefold().replace("ё", "е").split()).strip(" .!?…")
    return bool(re.fullmatch(
        r"(?:какие (?:параметры|настройки)(?: (?:нужно|надо|следует))?"
        r"(?: (?:задать|указать|выбрать|настроить|использовать))?"
        r"|что (?:нужно|надо|следует) настроить"
        r"|what (?:parameters|settings)(?: should i (?:set|use|choose))?)",
        text,
    ))


_HISTORY_LITERAL = re.compile(
    r"```.*?(?:```|$)|`[^`\n]*`|\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$|\$[^$\n]*\$"
    r"|\b\w+(?:\[[^\]\n]*\])+"
    r"|\[\s*[+-]?\d+(?:\.\d+)?(?:\s*,\s*[+-]?\d+(?:\.\d+)?)+\s*\]",
    re.DOTALL,
)


def _historical_prose(text: str) -> str:
    """Remove unambiguous old labels while preserving code, math and vectors."""

    pieces, offset = [], 0
    for literal in _HISTORY_LITERAL.finditer(text):
        pieces.append(re.sub(r"(?<![\w\]])\[[1-9]\d*\]", "", text[offset:literal.start()]))
        pieces.append(literal.group())
        offset = literal.end()
    pieces.append(re.sub(r"(?<![\w\]])\[[1-9]\d*\]", "", text[offset:]))
    return "".join(pieces)


def _model_history(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Retain conversation meaning without previous turns' citation namespaces."""

    latest_human = next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index].type == "human"),
        0,
    )
    history: list[AnyMessage] = []
    for message in messages[:latest_human]:
        if isinstance(message, ToolMessage) or (
            isinstance(message, AIMessage) and (message.tool_calls or message.invalid_tool_calls)
        ):
            continue
        if isinstance(message, AIMessage):
            message = message.model_copy(update={
                "content": "Исторический ответ (не источник; его ссылки относятся к прежнему вопросу):\n"
                + _historical_prose(str(message.content)),
                # Opaque reasoning belongs only to the tool loop that created
                # it. Retain the visible dialogue, not provider continuation data.
                "additional_kwargs": {}, "response_metadata": {},
            })
        history.append(message)
    return [*history, *messages[latest_human:]]
