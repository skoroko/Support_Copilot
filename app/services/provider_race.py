"""Bounded duplicate generation for slow providers; no tool execution here."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from app.observability.logging import get_logger

logger = get_logger("provider-race")


@dataclass
class BackupAllowance:
    used: int = 0
    limit: int = 1


backup_allowance: ContextVar[BackupAllowance | None] = ContextVar("generation_backup_allowance", default=None)


def _complete(message: AIMessage) -> bool:
    metadata = message.response_metadata
    return (metadata.get("status") in {None, "completed"}
            and metadata.get("finish_reason") not in {"length", "content_filter", "error"}
            and not message.invalid_tool_calls
            and bool(message.content or message.tool_calls))


def _terminal(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in {408, 409, 429}


async def first_complete(
    invoke: Callable[[], Awaitable[AIMessage]], *, delay: float,
    backup_slots: asyncio.Semaphore,
) -> AIMessage:
    """Race two identical invocations within a shared per-turn backup limit.

    Early failures retain the SDK's existing retry policy. Once a slow call
    has a sibling, a partial response cannot beat a complete response. Every
    exit drains both tasks; cancellation never leaves a client stream running.
    """
    tasks = [asyncio.create_task(invoke())]
    acquired = False
    winner = None
    race = None
    try:
        done, _ = await asyncio.wait(tasks, timeout=delay)
        if done or tasks[0].done():
            return await tasks[0]
        allowance = backup_allowance.get()
        if (allowance is not None and allowance.used >= allowance.limit) or backup_slots.locked():
            return await tasks[0]
        # No intervening await can claim the last slot on this event loop;
        # acquire is immediate. Unrelated turns never queue behind a backup.
        await backup_slots.acquire()
        acquired = True
        if tasks[0].done():
            return await tasks[0]
        if allowance is not None:
            allowance.used += 1
        tasks.append(asyncio.create_task(invoke()))
        pending = set(tasks)
        outcomes: dict[asyncio.Task, AIMessage | Exception] = {}
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            # Stable primary-first selection if both finish in one loop tick.
            for task in tasks:
                if task not in done:
                    continue
                try:
                    outcomes[task] = task.result()
                except Exception as error:  # noqa: BLE001 - preserve provider exceptions
                    outcomes[task] = error
            for outcome in outcomes.values():
                if isinstance(outcome, Exception) and _terminal(outcome):
                    raise outcome
            for task in tasks:
                outcome = outcomes.get(task)
                if isinstance(outcome, AIMessage) and _complete(outcome):
                    winner = task
                    break
            if winner is not None:
                break
        if winner is None:
            # Neither response is complete: keep the primary's established
            # failure/compaction behavior instead of manufacturing success.
            winner = tasks[0]
        result = await winner
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if acquired:
            backup_slots.release()
        if len(tasks) == 2:
            # Usage of cancelled/failed attempts is unknown, never zero cost.
            known_usage: list[dict[str, Any]] = []
            unknown_usage = 0
            for attempt, task in enumerate(tasks):
                if task.cancelled() or task.exception() is not None:
                    unknown_usage += 1
                    continue
                usage = task.result().usage_metadata
                if usage is None:
                    unknown_usage += 1
                else:
                    known_usage.append({"attempt": attempt, "usage": dict(usage)})
            race = {"attempts": len(tasks), "winner": tasks.index(winner) if winner else None,
                "known_usage": known_usage, "unknown_usage_attempts": unknown_usage}
            logger.info("agent_generation_backup_finished", **race)
    return result.model_copy(update={"response_metadata": {**result.response_metadata,
        "generation_race": race}}) if race else result
