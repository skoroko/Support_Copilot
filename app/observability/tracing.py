from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any, Iterator
from urllib.parse import urlparse

from app.observability.logging import get_logger
from app.observability.pii import prompt_hash


logger = get_logger("llm-service")
_tracing_configured = False


def _load_tracing_dependencies() -> tuple[Any, Any, Any]:
    """Import optional tracing packages inside the fail-open boundary."""

    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from phoenix.otel import register

    return LlamaIndexInstrumentor, OpenAIInstrumentor, register


def _tracing_enabled() -> bool:
    return os.environ.get("TRACING_ENABLED", "false").strip().casefold() in {
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
        LlamaIndexInstrumentor, OpenAIInstrumentor, register = _load_tracing_dependencies()
        tracer_provider: Any = register(project_name=project_name, endpoint=endpoint)
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception as exc:  # noqa: BLE001 - tracing must not break app startup
        logger.warning(
            "tracing_setup_failed",
            error_type=type(exc).__name__,
        )
        return False

    _tracing_configured = True
    logger.info("tracing_setup_completed", endpoint=endpoint, project_name=project_name)
    return True


@contextmanager
def trace_rag_query(question: str) -> Iterator[None]:
    """Create one parent span so retrieval and LLM calls form a single trace."""

    if not _tracing_configured:
        yield
        return

    try:
        from opentelemetry import trace
    except ImportError:  # pragma: no cover - optional tracing dependency
        yield
        return

    tracer = trace.get_tracer("support-copilot.rag")
    with tracer.start_as_current_span("RAGService.answer") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("support.question_hash", prompt_hash(question))
        span.set_attribute("support.question_chars", len(question))
        yield
