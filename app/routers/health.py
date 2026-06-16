from __future__ import annotations

import asyncio
import inspect
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.deps.providers import CacheDep


router = APIRouter(tags=["health"])


RESPONSES = {
    200: {"description": "Service process is alive"},
    422: {"description": "Validation error"},
    429: {"description": "LLM provider rate limit"},
    502: {"description": "LLM provider/auth error"},
    504: {"description": "LLM provider timeout"},
}

READY_RESPONSES = {
    200: {"description": "Service is ready to receive traffic"},
    503: {"description": "Service is alive but Redis is unavailable"},
}


async def _ping_cache(cache: Any) -> None:
    ping = getattr(cache, "ping", None)
    if ping is None:
        raise RuntimeError("cache backend does not expose ping")

    result = ping()
    if inspect.isawaitable(result):
        await result


@router.get(
    "/health",
    summary="Health check without external dependencies",
    responses=RESPONSES,
)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness check with Redis dependency",
    response_model=None,
    responses=READY_RESPONSES,
)
async def ready(cache: CacheDep) -> dict[str, str] | JSONResponse:
    try:
        await asyncio.wait_for(_ping_cache(cache), timeout=2.0)
    except Exception:
        return JSONResponse(status_code=503, content={"status": "degraded", "redis": "down"})

    return {"status": "ok", "redis": "up"}
