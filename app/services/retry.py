from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.core.exceptions import LLMRateLimitError, LLMTimeoutError


T = TypeVar("T")


RetryableError = LLMRateLimitError | LLMTimeoutError


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.25,
    max_delay_seconds: float = 4.0,
    jitter_seconds: float = 0.1,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except (LLMRateLimitError, LLMTimeoutError):
            if attempt >= max_attempts:
                raise
            delay = min(base_delay_seconds * (2 ** (attempt - 1)), max_delay_seconds)
            if jitter_seconds > 0:
                delay += random.random() * jitter_seconds
            await sleep(delay)

    raise RuntimeError("retry loop exhausted unexpectedly")
