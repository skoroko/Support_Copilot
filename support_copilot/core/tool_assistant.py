from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from support_copilot.infrastructure.tool_calling import ToolCallingClient, ToolCallingResult
from support_copilot.prompts.loader import render_system_prompt
from support_copilot.tools import TOOL_HANDLERS, available_tools


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JSONLToolLogger:
    def __init__(self, path: str | Path, *, reset: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_text("", encoding="utf-8")

    def write_event(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": utc_now(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def write(self, text: str) -> None:
        self.write_event("log", {"text": text})


class SupportToolAssistant:
    def __init__(
        self,
        client: ToolCallingClient,
        logger: JSONLToolLogger,
        *,
        product_name: str = "MIDAS Support Copilot",
        prompt_version: str = "v1",
        kb_version: str = "support-kb-v1",
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> None:
        self.client = client
        self.logger = logger
        self.product_name = product_name
        self.prompt_version = prompt_version
        self.kb_version = kb_version
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer(self, user_input: str) -> ToolCallingResult:
        self.logger.write_event("input", {"text": user_input})
        messages = [
            {
                "role": "system",
                "content": render_system_prompt(
                    self.prompt_version,
                    product_name=self.product_name,
                    kb_version=self.kb_version,
                ),
            },
            {"role": "user", "content": user_input},
        ]
        result = self.client.chat_with_tools(
            messages,
            tools=available_tools(),
            tool_handlers=TOOL_HANDLERS,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            event_sink=self.logger.write_event,
        )
        self.logger.write_event(
            "final_answer",
            {
                "provider": result.provider,
                "model": result.model,
                "direct_answer": result.direct_answer,
                "answer": result.answer,
            },
        )
        self.logger.write_event(
            "usage",
            {
                "provider": result.provider,
                "model": result.model,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "cached_tokens": result.usage.cached_tokens,
                "reasoning_tokens": result.usage.reasoning_tokens,
                "total_tokens": result.usage.total_tokens,
                "cost_usd": result.cost_usd,
            },
        )
        return result
