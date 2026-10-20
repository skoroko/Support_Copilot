"""One elapsed-time budget shared by an agent turn and its provider work."""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar

from app.services.provider_race import BackupAllowance, backup_allowance

_deadline: ContextVar[float | None] = ContextVar("agent_turn_deadline", default=None)


class AgentTurnTimeout(TimeoutError):
    """The complete turn, rather than one socket read, exhausted its budget."""


def remaining_seconds(limit: float) -> float:
    deadline = _deadline.get()
    remaining = limit if deadline is None else min(limit, deadline - time.perf_counter())
    if remaining <= 0:
        raise AgentTurnTimeout("Agent turn budget exhausted")
    return remaining


@asynccontextmanager
async def turn_budget(seconds: float, *, max_backups: int = 1):
    seconds = remaining_seconds(seconds)
    token = _deadline.set(time.perf_counter() + seconds)
    backup_token = (backup_allowance.set(BackupAllowance(limit=max(0, min(max_backups, 2))))
                    if backup_allowance.get() is None else None)
    try:
        timeout = asyncio.timeout(seconds)
        try:
            async with timeout:
                yield
        except TimeoutError as exc:
            if timeout.expired():
                raise AgentTurnTimeout("Agent turn budget exhausted") from exc
            raise
    finally:
        if backup_token is not None:
            backup_allowance.reset(backup_token)
        _deadline.reset(token)
