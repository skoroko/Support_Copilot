"""Server-sent-event keep-alive helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Final

HEARTBEAT_SECONDS: Final = 15.0
HEARTBEAT_COMMENT: Final = ": heartbeat\n\n"
_END = object()


async def with_heartbeats(
    events: AsyncIterator[str],
    interval: float = HEARTBEAT_SECONDS,
) -> AsyncIterator[str]:
    """Emit an SSE comment whenever the event source stays silent.

    Buffered turns (for example a persistent-agent answer that only yields at
    the end) otherwise leave proxies and clients with no bytes for tens of
    seconds, which trips read timeouts and idle connection drops. Comments are
    invisible to SSE parsers but reset transport-level read deadlines.
    """

    pending = asyncio.ensure_future(anext(events, _END))
    try:
        while True:
            done, _ = await asyncio.wait((pending,), timeout=interval)
            if pending not in done:
                yield HEARTBEAT_COMMENT
                continue
            event = pending.result()
            if event is _END:
                return
            yield event
            pending = asyncio.ensure_future(anext(events, _END))
    finally:
        if not pending.done():
            pending.cancel()
            with suppress(BaseException):
                await pending
        aclose = getattr(events, "aclose", None)
        if aclose is not None:
            with suppress(Exception):
                await aclose()
