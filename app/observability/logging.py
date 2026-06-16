from __future__ import annotations

import contextvars
import json
import logging as std_logging
from datetime import datetime, timezone
from typing import Any

try:
    import structlog
    from structlog.contextvars import bind_contextvars as _structlog_bind_contextvars
    from structlog.contextvars import clear_contextvars as _structlog_clear_contextvars
except ImportError:  # pragma: no cover - local tests can run without optional deps
    structlog = None
    _structlog_bind_contextvars = None
    _structlog_clear_contextvars = None


_fallback_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "fallback_log_context",
    default={},
)


class _FallbackJSONLogger:
    def __init__(self, name: str) -> None:
        self._logger = std_logging.getLogger(name)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(std_logging.INFO, event, kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(std_logging.WARNING, event, kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        kwargs["exc_info"] = True
        self._log(std_logging.ERROR, event, kwargs)

    def _log(self, level: int, event: str, fields: dict[str, Any]) -> None:
        exc_info = fields.pop("exc_info", None)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": std_logging.getLevelName(level).lower(),
            "event": event,
            **_fallback_context.get(),
            **fields,
        }
        self._logger.log(level, json.dumps(payload, ensure_ascii=False, default=str), exc_info=exc_info)


def setup_logging(level: str = "INFO") -> None:
    resolved_level = getattr(std_logging, level.upper(), std_logging.INFO)
    std_logging.basicConfig(level=resolved_level, format="%(message)s")
    for noisy_logger in ("httpx", "openai", "uvicorn.access"):
        std_logging.getLogger(noisy_logger).setLevel(std_logging.WARNING)

    if structlog is None:
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(resolved_level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    if structlog is not None:
        return structlog.get_logger(name)
    return _FallbackJSONLogger(name)


def bind_request_context(**fields: Any) -> None:
    clean_fields = {key: value for key, value in fields.items() if value is not None}
    if _structlog_bind_contextvars is not None:
        _structlog_bind_contextvars(**clean_fields)
        return

    current = dict(_fallback_context.get())
    current.update(clean_fields)
    _fallback_context.set(current)


def clear_request_context() -> None:
    if _structlog_clear_contextvars is not None:
        _structlog_clear_contextvars()
        return

    _fallback_context.set({})
