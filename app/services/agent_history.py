"""Project durable dialogue into a fresh, bounded model conversation."""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

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
