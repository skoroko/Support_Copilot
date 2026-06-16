from __future__ import annotations

import inspect
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings
from app.core.exceptions import LLMAuthError, LLMError, LLMRateLimitError, LLMTimeoutError
from app.routers import chat, health, models

try:
    from redis.asyncio import Redis
except ImportError:  # pragma: no cover - local tests can run without redis package
    Redis = None


logger = logging.getLogger("llm-service")
logging.basicConfig(level=logging.INFO)


class InMemoryAsyncCache:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._values.get(key)

    async def setex(self, key: str, _ttl_seconds: int, value: str) -> bool:
        self._values[key] = value
        return True

    async def close(self) -> None:
        self._values.clear()


def _build_cache(settings: Settings) -> Any:
    if Redis is None:
        return InMemoryAsyncCache()
    return Redis.from_url(settings.redis_url, decode_responses=True)


async def _maybe_close(obj: Any) -> None:
    close = getattr(obj, "close", None) or getattr(obj, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.openai = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
        timeout=settings.llm.request_timeout,
    )
    app.state.cache = _build_cache(settings)
    try:
        yield
    finally:
        await _maybe_close(app.state.openai)
        await _maybe_close(app.state.cache)


def _cors_origins() -> list[str]:
    try:
        return get_settings().cors_origins
    except Exception:
        return ["*"]


app = FastAPI(
    title="Support Copilot LLM Service",
    version="0.3.4",
    description="FastAPI wrapper for async Support Copilot LLM calls.",
    lifespan=lifespan,
)

cors_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "http.request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            status_code,
            duration_ms,
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status_code,
                "duration_ms": duration_ms,
            },
        )


@app.exception_handler(LLMError)
async def llm_error_handler(_request: Request, exc: LLMError) -> JSONResponse:
    if isinstance(exc, LLMRateLimitError):
        status_code = 429
    elif isinstance(exc, LLMTimeoutError):
        status_code = 504
    elif isinstance(exc, LLMAuthError):
        status_code = 502
    else:
        status_code = 502
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    details = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", []) if item != "body")
        details.append({"field": location or "body", "message": error.get("msg", "Invalid value")})
    return JSONResponse(status_code=422, content={"error": {"code": "validation_error", "details": details}})


app.include_router(health.router)
app.include_router(models.router)
app.include_router(chat.router)
