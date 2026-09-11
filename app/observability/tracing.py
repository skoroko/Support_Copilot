from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from app.observability.logging import get_logger


logger = get_logger("llm-service")
_tracing_configured = False


def _tracing_enabled() -> bool:
    return os.environ.get("TRACING_ENABLED", "true").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _normalize_phoenix_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.path in {"", "/"}:
        return endpoint.rstrip("/") + "/v1/traces"
    return endpoint


def setup_tracing(project_name: str | None = None) -> bool:
    global _tracing_configured
    if _tracing_configured:
        return True
    if not _tracing_enabled():
        logger.info("tracing_setup_skipped", reason="disabled")
        return False

    endpoint = _normalize_phoenix_endpoint(os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"))
    project_name = project_name or os.environ.get("PHOENIX_PROJECT_NAME", "support-copilot-rag")
    try:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from phoenix.otel import register
    except ImportError as exc:  # pragma: no cover - optional local dependency
        logger.warning(
            "tracing_setup_skipped",
            reason="missing_dependency",
            dependency_error=type(exc).__name__,
        )
        return False

    try:
        tracer_provider: Any = register(project_name=project_name, endpoint=endpoint)
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception as exc:  # noqa: BLE001 - tracing must not break app startup
        logger.warning(
            "tracing_setup_failed",
            endpoint=endpoint,
            error_type=type(exc).__name__,
        )
        return False

    _tracing_configured = True
    logger.info("tracing_setup_completed", endpoint=endpoint, project_name=project_name)
    return True
