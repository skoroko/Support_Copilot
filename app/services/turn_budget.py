"""One elapsed-time budget shared by an agent turn and its provider work."""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar

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
async def turn_budget(seconds: float):
    seconds = remaining_seconds(seconds)
    token = _deadline.set(time.perf_counter() + seconds)
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
        _deadline.reset(token)
