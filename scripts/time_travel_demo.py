"""Offline demonstration of durable LangGraph approval, history and replay.

Run from the repository root:
``.venv-eval\Scripts\python.exe scripts\time_travel_demo.py``
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Make ``python scripts/time_travel_demo.py`` work from any current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.agent_persistent import APPROVAL_NODE, Command, build_agent


class DemoModel:
    """Deterministic offline model: request Telegram, then acknowledge outcome."""

    async def ainvoke(self, messages, **_kwargs) -> AIMessage:
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="Решение зафиксировано в учебном графе.")
        return AIMessage(
            content="Подготовлю уведомление для подтверждения.",
            tool_calls=[
                {
                    "name": "send_telegram_message",
                    "args": {"chat_id": "4242", "text": "Инцидент передан инженеру."},
                    "id": "demo-telegram-call",
                }
            ],
        )


def config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id, "user_role": "write-with-approve"}}


def input_state() -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content="Передай инженеру, что требуется проверка лицензии.")],
        "tool_results": [],
    }


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "value"):
        return {"value": json_safe(value.value)}
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    return str(value)


async def run_demo() -> None:
    deliveries: list[tuple[str, str]] = []

    async def send_message(chat_id: str, text: str, idempotency_key: str) -> str:
        del idempotency_key
        deliveries.append((chat_id, text))
        return f"demo delivery to {chat_id}"

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(saver, model=DemoModel(), send_message=send_message)

        history_config = config("demo-history")
        paused = await graph.ainvoke(input_state(), history_config)
        print("INTERRUPT", json.dumps(json_safe(paused["__interrupt__"]), ensure_ascii=False))

        history = [snapshot async for snapshot in graph.aget_state_history(history_config)]
        print("HISTORY checkpoint_id next")
        for snapshot in history[:5]:
            checkpoint_id = snapshot.config["configurable"].get("checkpoint_id", "-")
            print(checkpoint_id, ",".join(snapshot.next) or "END")

        before_send = next(
            snapshot
            for snapshot in history
            if snapshot.values.get("telegram_draft") and not snapshot.values.get("sent") and APPROVAL_NODE in snapshot.next
        )
        old_id = before_send.config["configurable"]["checkpoint_id"]
        past = await graph.aget_state({"configurable": {"thread_id": "demo-history", "checkpoint_id": old_id}})
        print(
            "PAST",
            json.dumps(
                {
                    "checkpoint_id": old_id,
                    "draft": past.values.get("telegram_draft"),
                    "sent": past.values.get("sent"),
                    "next": list(past.next),
                },
                ensure_ascii=False,
            ),
        )

        # Command(resume=...) is deterministic within one lineage.  Separate,
        # stable thread ids intentionally form two valid alternatives from the
        # same original input rather than attempting to replay one decision.
        reject_config = config("demo-reject")
        await graph.ainvoke(input_state(), reject_config)
        rejected = await graph.ainvoke(Command(resume=False), reject_config)
        print("REJECT", json.dumps({"sent": rejected["sent"], "delivery_count": len(deliveries)}, ensure_ascii=False))

        approve_config = config("demo-approve")
        await graph.ainvoke(input_state(), approve_config)
        approved = await graph.ainvoke(Command(resume=True), approve_config)
        print("APPROVE", json.dumps({"sent": approved["sent"], "delivery_count": len(deliveries)}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(run_demo())
