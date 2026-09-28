from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

try:
    from opentelemetry.trace import Span as _OpenTelemetrySpan
except ImportError:  # Optional tracing packages must not prevent app startup.
    _OpenTelemetrySpan = object

from app.observability.logging import get_logger
from app.observability.pii import prompt_hash

logger = get_logger("llm-service")
_tracing_configured = False


def _load_tracing_dependencies() -> tuple[Any, Any, Any, Any]:
    """Import optional tracing packages inside the fail-open boundary."""

    from openinference.instrumentation import TraceConfig
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from phoenix.otel import register

    return TraceConfig, LlamaIndexInstrumentor, OpenAIInstrumentor, register


def _private_span_attribute(key: str) -> bool:
    return (
        key == "retrieval.documents"
        or key.startswith(("retrieval.documents.", "exception."))
        or key == "exception"
        or key in {"error.message", "error.stack", "error.stacktrace"}
    )


def _privacy_trace_config(trace_config: Any) -> Any:
    """Require content masking before any optional instrumentor starts."""

    if not callable(getattr(trace_config, "mask", None)):
        raise TypeError("Tracing package does not support attribute masking")

    class PrivateTraceConfig(trace_config):
        def mask(self, key: str, value: Any, *, externalize: bool = True) -> Any:
            # OpenInference 0.1.65 does not classify retrieved documents as inputs.
            if _private_span_attribute(key):
                return None
            return super().mask(key, value, externalize=externalize)

    return PrivateTraceConfig(
        hide_llm_invocation_parameters=True,
        hide_llm_tools=True,
        hide_inputs=True,
        hide_outputs=True,
        hide_input_messages=True,
        hide_output_messages=True,
        hide_input_images=True,
        hide_input_text=True,
        hide_output_text=True,
        hide_embedding_vectors=True,
        hide_embeddings_vectors=True,
        hide_embeddings_text=True,
        hide_prompts=True,
        hide_choices=True,
        enable_genai_semconv=False,
    )


class _PrivateSpan(_OpenTelemetrySpan):
    """Keep OTel parent context while suppressing content outside TraceConfig."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def __getattr__(self, name: str) -> Any:
        return getattr(self._span, name)

    def end(self, end_time: int | None = None) -> None:
        self._span.end(end_time=end_time)

    def get_span_context(self) -> Any:
        return self._span.get_span_context()

    def is_recording(self) -> bool:
        return self._span.is_recording()

    def update_name(self, name: str) -> None:
        self._span.update_name(name)

    def add_link(self, context: Any, attributes: Any = None) -> None:
        # Link attributes are free-form and do not pass through TraceConfig.
        self._span.add_link(context)

    def set_attribute(self, key: str, value: Any) -> None:
        if _private_span_attribute(key):
            return
        self._span.set_attribute(key, value)

    def set_attributes(self, attributes: Any) -> None:
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def add_event(self, name: str, attributes: Any = None, timestamp: int | None = None) -> None:
        # Event names and attributes are free-form and may contain raw content.
        return

    def record_exception(self, exception: BaseException, attributes: Any = None, timestamp: int | None = None, escaped: bool = False) -> None:
        return

    def set_status(self, status: Any, description: str | None = None) -> None:
        # LlamaIndex puts the exception message in Status.description.
        if hasattr(status, "status_code"):
            status = type(status)(status_code=status.status_code)
        self._span.set_status(status)


class _PrivateTracer:
    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tracer, name)

    def start_span(self, *args: Any, **kwargs: Any) -> _PrivateSpan:
        return _PrivateSpan(self._tracer.start_span(*args, **kwargs))


class _PrivateTracerProvider:
    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def get_tracer(self, *args: Any, **kwargs: Any) -> _PrivateTracer:
        return _PrivateTracer(self._provider.get_tracer(*args, **kwargs))


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
        TraceConfig, LlamaIndexInstrumentor, OpenAIInstrumentor, register = _load_tracing_dependencies()
        privacy_config = _privacy_trace_config(TraceConfig)
        tracer_provider = _PrivateTracerProvider(register(project_name=project_name, endpoint=endpoint))
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=privacy_config)
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider, config=privacy_config)
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
    with tracer.start_as_current_span(
        "RAGService.answer", record_exception=False, set_status_on_exception=False
    ) as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("support.question_hash", prompt_hash(question))
        span.set_attribute("support.question_chars", len(question))
        yield
