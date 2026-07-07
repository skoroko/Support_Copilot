from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)\s*```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ParsedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


def _read_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def parse_json_response(text: str) -> dict[str, Any]:
    candidate = _extract_json_candidate(text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON response: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("JSON response must be an object")
    return payload


def parse_tool_calls(message: Any) -> list[ParsedToolCall]:
    calls = _read_field(message, "tool_calls", []) or []
    parsed: list[ParsedToolCall] = []

    for index, call in enumerate(calls):
        function = _read_field(call, "function", {}) or {}
        name = str(_read_field(function, "name", "") or "")
        if not name:
            raise ValueError("Tool call function name is required")

        raw_arguments = _read_field(function, "arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed tool arguments for {name}: {exc.msg}") from exc
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            raise ValueError(f"Tool arguments for {name} must be a JSON object")

        if not isinstance(arguments, dict):
            raise ValueError(f"Tool arguments for {name} must be a JSON object")

        parsed.append(
            ParsedToolCall(
                id=str(_read_field(call, "id", None) or f"call_{index + 1}"),
                name=name,
                arguments=arguments,
            )
        )

    return parsed


def _extract_json_candidate(text: str) -> str:
    stripped = text.strip()
    match = JSON_FENCE_RE.search(stripped)
    if match:
        return match.group("body").strip()
    return stripped
