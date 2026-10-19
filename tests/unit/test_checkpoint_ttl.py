from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.agent_persistent import (
    DEFAULT_CHECKPOINT_TTL_HOURS,
    _checkpoint_ttl_hours,
    _delete_stale_checkpoint_threads,
)


def _record(thread_id: str, ts: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        config={"configurable": {"thread_id": thread_id}},
        checkpoint={"ts": ts} if ts is not None else {},
        metadata={},
    )


def _saver(records: list[SimpleNamespace]) -> SimpleNamespace:
    async def alist(_config):
        for record in records:
            yield record

    return SimpleNamespace(alist=alist, adelete_thread=AsyncMock())


@pytest.mark.asyncio
async def test_stale_threads_are_deleted_while_recent_and_mixed_age_survive() -> None:
    now = datetime.now(timezone.utc)
    old = (now - timedelta(hours=DEFAULT_CHECKPOINT_TTL_HOURS + 1)).isoformat()
    fresh = (now - timedelta(hours=1)).isoformat()
    records = [
        _record("old-thread", old),
        _record("old-thread", old),
        _record("fresh-thread", old),
        _record("fresh-thread", fresh),
        _record("broken-thread", "not-a-date"),
    ]
    saver = _saver(records)

    removed = await _delete_stale_checkpoint_threads(
        saver, ttl_hours=DEFAULT_CHECKPOINT_TTL_HOURS
    )

    assert removed == 1
    saver.adelete_thread.assert_awaited_once_with("old-thread")


@pytest.mark.asyncio
async def test_ttl_hours_env_zero_disables_pruning(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_CHECKPOINT_TTL_HOURS", "0")
    assert _checkpoint_ttl_hours() is None
    monkeypatch.setenv("AGENT_CHECKPOINT_TTL_HOURS", "48")
    assert _checkpoint_ttl_hours() == 48.0
    monkeypatch.delenv("AGENT_CHECKPOINT_TTL_HOURS")
    assert _checkpoint_ttl_hours() == DEFAULT_CHECKPOINT_TTL_HOURS
