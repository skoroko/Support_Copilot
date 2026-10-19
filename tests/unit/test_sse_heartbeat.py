from __future__ import annotations

import asyncio

import pytest

from app.core.sse import HEARTBEAT_COMMENT, with_heartbeats


async def _collect(source, interval: float = 0.02) -> list[str]:
    return [chunk async for chunk in with_heartbeats(source, interval=interval)]


@pytest.mark.asyncio
async def test_heartbeats_fill_silence_and_events_stay_in_order() -> None:
    async def events():
        yield "data: one\n\n"
        await asyncio.sleep(0.09)
        yield "data: two\n\n"

    output = await _collect(events())

    assert output[0] == "data: one\n\n"
    assert output[-1] == "data: two\n\n"
    heartbeats = [chunk for chunk in output[1:-1] if chunk == HEARTBEAT_COMMENT]
    assert len(heartbeats) >= 2


@pytest.mark.asyncio
async def test_fast_source_emits_no_heartbeats() -> None:
    async def events():
        yield "data: one\n\n"
        yield "data: two\n\n"

    assert await _collect(events()) == ["data: one\n\n", "data: two\n\n"]


@pytest.mark.asyncio
async def test_source_error_propagates_without_hanging() -> None:
    async def events():
        yield "data: one\n\n"
        raise RuntimeError("stream failed")
        yield  # pragma: no cover - keep it an async generator

    with pytest.raises(RuntimeError, match="stream failed"):
        await _collect(events())


@pytest.mark.asyncio
async def test_early_close_cancels_pending_read_and_closes_source() -> None:
    closed = asyncio.Event()

    async def events():
        try:
            await asyncio.Event().wait()
            yield "data: never\n\n"
        finally:
            closed.set()

    stream = with_heartbeats(events(), interval=0.02)
    assert await anext(stream) == HEARTBEAT_COMMENT
    await stream.aclose()

    assert closed.is_set()
