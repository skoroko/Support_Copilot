from __future__ import annotations

import asyncio
import inspect
from typing import Any

from fastapi import APIRouter, Request
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
    200: {"description": "Service is ready to receive traffic, possibly degraded"},
    503: {"description": "Service is alive but Redis is unavailable"},
}


async def _ping_cache(cache: Any) -> None:
    ping = getattr(cache, "ping", None)
    if ping is None:
        raise RuntimeError("cache backend does not expose ping")

    result = ping()
    if inspect.isawaitable(result):
        await result


def _rag_readiness(request: Request) -> tuple[str, str | None]:
    settings = getattr(request.app.state, "settings", None)
    if not getattr(settings, "rag_enabled", False):
        return "disabled", None
    if getattr(request.app.state, "rag_service", None) is not None:
        return "up", None
    error = getattr(request.app.state, "rag_startup_error", None)
    return "down", str(error) if error else "unknown"


def _ready_payload(redis: str, rag: str, rag_error: str | None) -> dict[str, Any]:
    degraded = redis == "down" or rag == "down"
    content: dict[str, Any] = {
        "status": "degraded" if degraded else "ok",
        "redis": redis,
        "rag": rag,
    }
    if rag_error:
        content["rag_error"] = rag_error
    return content


@router.get(
    "/health",
    summary="Health check without external dependencies",
    responses=RESPONSES,
)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness check with Redis and RAG startup status",
    response_model=None,
    responses=READY_RESPONSES,
)
async def ready(request: Request, cache: CacheDep) -> dict[str, Any] | JSONResponse:
    rag, rag_error = _rag_readiness(request)
    try:
        await asyncio.wait_for(_ping_cache(cache), timeout=2.0)
    except Exception:
        return JSONResponse(status_code=503, content=_ready_payload("down", rag, rag_error))

    # RAG being down degrades answers but must not restart-loop the container:
    # the compose healthcheck targets this endpoint, so keep 200 and flag it.
    return _ready_payload("up", rag, rag_error)
