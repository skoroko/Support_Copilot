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
from fastapi.concurrency import run_in_threadpool
from openai import AsyncOpenAI

from app.admin import routes as admin_routes
from app.chat import routes as chat_routes
from app.chat.service import count_tokens
from app.core.config import Settings, get_settings
from app.core.auth import InMemoryRequestBudget
from app.core.upload_boundary import UploadBoundaryMiddleware
from app.core.exceptions import LLMAuthError, LLMError, LLMOutputBlockedError, LLMRateLimitError, LLMTimeoutError
from app.observability.logging import bind_request_context, clear_request_context, get_logger, setup_logging
from app.observability.tracing import setup_tracing
from app.routers import agent, chat, documents, health, models, rag
from app.services.agent_persistent import agent_lifespan
from app.services.rag import RAGConfig, RAGService, build_llm_translate_hook
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


def _initialize_rag(settings: Settings) -> tuple[RAGService | None, str | None]:
    """Build RAG without making an optional provider outage fatal to the API."""

    config = RAGConfig.from_settings(settings)
    if not getattr(config, "allow_external_data_transfer", True):
        # Without the opt-in the built-in embedding adapter raises RuntimeError
        # on every question; surface the misconfiguration once here instead.
        error_type = "ExternalDataTransferNotAllowed"
        logger.warning("rag_startup_failed", error_type=error_type)
        return None, error_type
    # Retrieval calls without a caller-supplied English variant (the agent's
    # evidence prefetch, plain answer()/retrieve() turns) translate the
    # effective question once so English manuals are covered; a hook failure
    # degrades to the Russian-only path.
    translate_hook = (
        build_llm_translate_hook(config)
        if getattr(config, "translate_llm_enabled", True)
        else None
    )
    service = RAGService(config, translate_hook=translate_hook, own_translate_hook=True)
    try:
        service.build()
    except Exception as exc:
        error_type = type(exc).__name__
        logger.warning("rag_startup_failed", error_type=error_type)
        service.close()
        return None, error_type
    return service, None


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
    app.state.request_budget = InMemoryRequestBudget(
        requests_per_minute=settings.api_rate_limit_per_minute,
        max_concurrency=settings.api_max_concurrency_per_principal,
    )
    app.state.canary = f"CANARY_{secrets.token_hex(4)}"
    app.state.security_system_prompt = build_security_system_prompt(app.state.canary)
    app.state.openai = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
        timeout=settings.llm.request_timeout,
    )
    app.state.cache = _build_cache(settings)
    app.state.rag_service = None
    app.state.ingestion_service = None
    app.state.rag_startup_error = None
    try:
        async with agent_lifespan(app):
            # Load tokenizer data before readiness, outside the event loop.
            await run_in_threadpool(count_tokens, [{"role": "user", "content": "warmup"}])
            # The application owns exactly one vector store: the one inside
            # RAGService (RAG_COLLECTION).  The homework-5.2 VectorStore
            # adapter remains script-only so the experiment ``documents``
            # collection cannot drift out of sync with the production corpus
            # (audit F20) — and Qdrant is not contacted at startup at all
            # while RAG is disabled.
            if settings.rag_enabled:
                app.state.rag_service, app.state.rag_startup_error = await run_in_threadpool(
                    _initialize_rag,
                    settings,
                )
                if app.state.rag_service is not None:
                    app.state.ingestion_service = app.state.rag_service.ingestion_service
            yield
    finally:
        await _maybe_close(app.state.openai)
        await _maybe_close(app.state.cache)
        if app.state.rag_service is not None:
            await run_in_threadpool(app.state.rag_service.close)
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
    version="0.5.5",
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
app.add_middleware(UploadBoundaryMiddleware)


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
app.include_router(agent.router)
app.include_router(chat.router)
app.include_router(rag.router)
app.include_router(documents.router)
app.include_router(admin_routes.router)
app.include_router(chat_routes.router)
