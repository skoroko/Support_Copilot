from __future__ import annotations

import inspect
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from app.chat import routes as chat_routes
from app.core.config import Settings, get_settings
from app.core.exceptions import LLMAuthError, LLMError, LLMOutputBlockedError, LLMRateLimitError, LLMTimeoutError
from app.observability.logging import bind_request_context, clear_request_context, get_logger, setup_logging
from app.observability.tracing import setup_tracing
from app.routers import chat, health, models
from app.services.security.output_filter import build_security_system_prompt

try:
    from redis.asyncio import Redis
except ImportError:  # pragma: no cover - local tests can run without redis package
    Redis = None


logger = get_logger("llm-service")


class InMemoryAsyncCache:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._values.get(key)

    async def setex(self, key: str, _ttl_seconds: int, value: str) -> bool:
        self._values[key] = value
        return True

    async def ping(self) -> bool:
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
    setup_logging(settings.log_level)
    setup_tracing()
    app.state.settings = settings
    app.state.canary = f"CANARY_{secrets.token_hex(4)}"
    app.state.security_system_prompt = build_security_system_prompt(app.state.canary)
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
        chat_engine = getattr(app.state, "chat_engine", None)
        if chat_engine is not None:
            await chat_engine.dispose()


def _cors_origins() -> list[str]:
    try:
        return get_settings().cors_origins
    except Exception:
        return ["*"]


app = FastAPI(
    title="Support Copilot LLM Service",
    version="0.4.3",
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
    clear_request_context()
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    user_id = request.headers.get("X-User-ID")
    request.state.request_id = request_id
    bind_request_context(
        request_id=request_id,
        user_id=user_id,
        path=request.url.path,
        method=request.method,
    )
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "http_request_completed",
            status_code=status_code,
            latency_ms=round(duration_ms, 2),
        )
        clear_request_context()


@app.exception_handler(LLMError)
async def llm_error_handler(_request: Request, exc: LLMError) -> JSONResponse:
    if isinstance(exc, LLMRateLimitError):
        status_code = 429
    elif isinstance(exc, LLMTimeoutError):
        status_code = 504
    elif isinstance(exc, LLMAuthError):
        status_code = 502
    elif isinstance(exc, LLMOutputBlockedError):
        status_code = 400
    else:
        status_code = 502
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        content = {"error": exc.detail}
    else:
        content = {"error": {"code": "http_error", "message": str(exc.detail)}}
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


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
app.include_router(chat_routes.router)
