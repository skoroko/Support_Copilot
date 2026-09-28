from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from pydantic import SecretStr

from app.core.config import Settings
from app.observability.pii import prompt_hash, redact_pii
from app.observability.tracing import (
    _normalize_phoenix_endpoint,
    _privacy_trace_config,
    _PrivateTracerProvider,
    _tracing_enabled,
    setup_tracing,
    trace_rag_query,
)
from app.schemas.chat import ChatRequest, ChatResponse, Usage
from app.services.llm import LLMService


class CaptureLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def info(self, event: str, **kwargs) -> None:
        self.events.append((event, kwargs))


def build_settings() -> Settings:
    return Settings(
        openai_api_key=SecretStr("sk-test"),
        default_model="gpt-5.4-mini",
        request_timeout=30,
        redis_url="redis://redis:6379/0",
        cache_ttl_seconds=3600,
        cors_origins=["*"],
        log_level="INFO",
    )


class ObservabilityTests(unittest.TestCase):
    def test_redact_pii_replaces_email_phone_card_inn_and_passport(self) -> None:
        raw = (
            "Мой email ivan@mail.ru, тел +7 (999) 123-45-67, "
            "карта 4111 1111 1111 1111, ИНН 7707083893, паспорт 45 10 123456"
        )

        redacted = redact_pii(raw)

        for sensitive in [
            "ivan@mail.ru",
            "+7 (999) 123-45-67",
            "4111 1111 1111 1111",
            "7707083893",
            "45 10 123456",
        ]:
            self.assertNotIn(sensitive, redacted)
        for placeholder in ["[EMAIL]", "[PHONE_RU]", "[CARD]", "[INN]", "[PASSPORT]"]:
            self.assertIn(placeholder, redacted)

    def test_redact_pii_handles_self_check_sample(self) -> None:
        redacted = redact_pii("email@example.com, +7 999 123 45 67, order HD-123456")

        self.assertNotIn("email@example.com", redacted)
        self.assertNotIn("+7 999 123 45 67", redacted)
        self.assertNotIn("HD-123456", redacted)
        self.assertIn("[EMAIL]", redacted)
        self.assertIn("[PHONE_RU]", redacted)
        self.assertIn("[ORDER_ID]", redacted)

    def test_prompt_hash_is_stable_and_short(self) -> None:
        first = prompt_hash("same prompt")
        second = prompt_hash("same prompt")

        self.assertEqual(first, second)
        self.assertRegex(first, r"^sha256:[0-9a-f]{16}$")

    def test_phoenix_endpoint_normalization_adds_otlp_path(self) -> None:
        self.assertEqual(
            _normalize_phoenix_endpoint("http://phoenix:6006"),
            "http://phoenix:6006/v1/traces",
        )

    def test_tracing_can_be_disabled_explicitly(self) -> None:
        with patch.dict("os.environ", {"TRACING_ENABLED": "false"}):
            self.assertFalse(_tracing_enabled())
        self.assertEqual(
            _normalize_phoenix_endpoint("http://phoenix:6006/v1/traces"),
            "http://phoenix:6006/v1/traces",
        )

    def test_tracing_permission_error_is_fail_open(self) -> None:
        with (
            patch("app.observability.tracing._tracing_configured", False),
            patch("app.observability.tracing._tracing_enabled", return_value=True),
            patch("app.observability.tracing._load_tracing_dependencies", side_effect=PermissionError),
        ):
            self.assertFalse(setup_tracing())

    def test_tracing_os_error_is_fail_open(self) -> None:
        with (
            patch("app.observability.tracing._tracing_configured", False),
            patch("app.observability.tracing._tracing_enabled", return_value=True),
            patch("app.observability.tracing._load_tracing_dependencies", side_effect=OSError),
        ):
            self.assertFalse(setup_tracing())

    def test_tracing_applies_one_explicit_privacy_config_to_both_instrumentors(self) -> None:
        created = []

        class RecordingTraceConfig:
            def __init__(self, **options) -> None:
                self.options = options
                created.append(self)

            def mask(self, key, value, *, externalize=True):
                return value

        openai_instrumentor = Mock()
        llama_instrumentor = Mock()
        provider = Mock()
        register = Mock(return_value=provider)
        with (
            patch("app.observability.tracing._tracing_configured", False),
            patch("app.observability.tracing._tracing_enabled", return_value=True),
            patch(
                "app.observability.tracing._load_tracing_dependencies",
                return_value=(
                    RecordingTraceConfig,
                    Mock(return_value=llama_instrumentor),
                    Mock(return_value=openai_instrumentor),
                    register,
                ),
            ),
        ):
            self.assertTrue(setup_tracing(project_name="privacy-test"))

        self.assertEqual(len(created), 1)
        options = created[0].options
        self.assertTrue(all(value is True for name, value in options.items() if name != "enable_genai_semconv"))
        self.assertFalse(options["enable_genai_semconv"])
        self.assertTrue({
            "hide_inputs", "hide_outputs", "hide_embeddings_text", "hide_embeddings_vectors",
            "hide_input_images", "hide_prompts", "hide_choices", "hide_llm_invocation_parameters",
        } <= options.keys())
        register.assert_called_once()
        openai_provider = openai_instrumentor.instrument.call_args.kwargs["tracer_provider"]
        llama_provider = llama_instrumentor.instrument.call_args.kwargs["tracer_provider"]
        self.assertIsInstance(openai_provider, _PrivateTracerProvider)
        self.assertIs(openai_provider, llama_provider)
        self.assertIs(openai_provider._provider, provider)
        self.assertIs(openai_instrumentor.instrument.call_args.kwargs["config"], created[0])
        self.assertIs(llama_instrumentor.instrument.call_args.kwargs["config"], created[0])

    def test_tracing_does_not_register_or_instrument_without_compatible_privacy_config(self) -> None:
        class IncompatibleTraceConfig:
            def __init__(self) -> None:
                pass

        openai_factory = Mock()
        llama_factory = Mock()
        register = Mock()
        with (
            patch("app.observability.tracing._tracing_configured", False),
            patch("app.observability.tracing._tracing_enabled", return_value=True),
            patch(
                "app.observability.tracing._load_tracing_dependencies",
                return_value=(IncompatibleTraceConfig, llama_factory, openai_factory, register),
            ),
        ):
            self.assertFalse(setup_tracing())

        register.assert_not_called()
        openai_factory.assert_not_called()
        llama_factory.assert_not_called()

    def test_real_trace_config_masks_synthetic_child_span_content_when_installed(self) -> None:
        try:
            from openinference.instrumentation import TraceConfig
            from openinference.semconv.trace import EmbeddingAttributes, SpanAttributes
        except ImportError:
            self.skipTest("optional OpenInference tracing extra is not installed")

        marker = "synthetic-private-marker-4821"
        config = _privacy_trace_config(TraceConfig)
        sensitive_keys = (
            SpanAttributes.INPUT_VALUE,
            SpanAttributes.OUTPUT_VALUE,
            SpanAttributes.LLM_PROMPTS,
            SpanAttributes.LLM_CHOICES,
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}",
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}",
            "retrieval.documents.0.document.content",
            "retrieval.documents.0.document.metadata",
            "retrieval.documents.0.document.id",
            "retrieval.documents.0.document.score",
        )
        for key in sensitive_keys:
            self.assertNotIn(marker, str(config.mask(key, marker)))

    def test_private_trace_config_drops_all_retrieval_document_attributes(self) -> None:
        class PassThroughTraceConfig:
            def __init__(self, **options) -> None:
                self.options = options

            def mask(self, key, value, *, externalize=True):
                return value

        config = _privacy_trace_config(PassThroughTraceConfig)
        for suffix in ("content", "metadata", "id", "score", "custom"):
            self.assertIsNone(config.mask(f"retrieval.documents.0.document.{suffix}", "synthetic-marker"))
        self.assertIsNone(config.mask("retrieval.documents", "synthetic-marker"))
        self.assertEqual(config.mask("retrieval.documents_extra", 17), 17)

    def test_private_span_export_drops_retrieval_events_and_exception_text(self) -> None:
        marker = "synthetic-private-marker-4821"
        raw_span = Mock()
        raw_span.attributes = {}
        raw_span.events = []
        raw_span.status = None
        raw_span.set_attribute.side_effect = lambda key, value: raw_span.attributes.update({key: value})
        raw_span.add_event.side_effect = lambda name, *args, **kwargs: raw_span.events.append(name)
        raw_span.set_status.side_effect = lambda status, *args, **kwargs: setattr(raw_span, "status", status)
        raw_tracer = Mock()
        raw_tracer.start_span.return_value = raw_span
        provider = Mock()
        provider.get_tracer.return_value = raw_tracer
        span = _PrivateTracerProvider(provider).get_tracer("test").start_span("retriever")
        span.set_attributes({
            "retrieval.documents.0.document.content": marker,
            "retrieval.documents.0.document.metadata": marker,
            "llm.token_count.total": 3,
        })
        span.add_event(marker, {"text": marker})
        span.record_exception(ValueError(marker))

        class Status:
            def __init__(self, status_code, description=None) -> None:
                self.status_code = status_code
                self.description = description

        span.set_status(Status("ERROR", marker))
        self.assertEqual(raw_span.attributes, {"llm.token_count.total": 3})
        self.assertEqual(raw_span.events, [])
        self.assertIsNone(raw_span.status.description)
        raw_span.record_exception.assert_not_called()

    def test_real_llama_retriever_span_exports_no_synthetic_content_when_installed(self) -> None:
        try:
            from llama_index.core.schema import NodeWithScore, TextNode
            from openinference.instrumentation import OITracer, TraceConfig
            from openinference.instrumentation.llama_index._handler import _Span
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
        except ImportError:
            self.skipTest("optional OpenInference tracing extra is not installed")

        marker = "synthetic-private-marker-4821"
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = OITracer(
            _PrivateTracerProvider(provider).get_tracer("privacy-test"),
            config=_privacy_trace_config(TraceConfig),
        )
        span = _Span(otel_span=tracer.start_span("retriever"))
        span._process_nodes(
            "retrieval.documents",
            NodeWithScore(node=TextNode(text=marker, metadata={"private": marker}), score=0.5),
        )
        span.record_exception(ValueError(marker))
        span.end(exception=ValueError(marker))

        exported = exporter.get_finished_spans()
        self.assertEqual(len(exported), 1)
        child = exported[0]
        self.assertFalse(any(key.startswith("retrieval.documents") for key in child.attributes))
        self.assertNotIn(marker, str(child.attributes))
        self.assertEqual(child.events, ())
        self.assertNotIn(marker, str(child.status.description))

    def test_real_oi_context_span_drops_exception_details_when_installed(self) -> None:
        try:
            from openinference.instrumentation import OITracer, TraceConfig
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
        except ImportError:
            self.skipTest("optional OpenInference tracing extra is not installed")

        marker = "synthetic-private-marker-4821"
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = OITracer(
            _PrivateTracerProvider(provider).get_tracer("privacy-test"),
            config=_privacy_trace_config(TraceConfig),
        )
        with self.assertRaisesRegex(ValueError, marker), tracer.start_as_current_span("child") as span:
            span.add_event(marker, {"private": marker})
            raise ValueError(marker)

        child = exporter.get_finished_spans()[0]
        self.assertEqual(child.events, ())
        self.assertNotIn(marker, str(child.status.description))

    def test_real_instrumented_span_preserves_parent_context_and_privacy(self) -> None:
        try:
            from openinference.instrumentation import OITracer, TraceConfig
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
            from opentelemetry.trace import Span, Status, StatusCode
        except ImportError:
            self.skipTest("optional OpenInference tracing extra is not installed")

        marker = "synthetic-private-marker-4821"
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        raw_tracer = provider.get_tracer("privacy-test")
        instrumented_tracer = OITracer(
            _PrivateTracerProvider(provider).get_tracer("privacy-test"),
            config=_privacy_trace_config(TraceConfig),
        )
        with (
            patch("app.observability.tracing._tracing_configured", True),
            patch("opentelemetry.trace.get_tracer", return_value=raw_tracer),
            trace_rag_query(marker),
        ):
            root_context = trace.get_current_span().get_span_context()
            with instrumented_tracer.start_as_current_span("instrumented-child") as child:
                self.assertIsInstance(child, Span)
                child_context = trace.get_current_span().get_span_context()
                self.assertTrue(child_context.is_valid)
                self.assertEqual(child_context.span_id, child.get_span_context().span_id)
                self.assertEqual(child_context.trace_id, root_context.trace_id)
                child.set_attribute("retrieval.documents.0.document.content", marker)
                child.set_attribute("input.value", marker)
                child.add_event(marker, {"private": marker})
                child.add_link(root_context, {"private": marker})
                child.record_exception(ValueError(marker))
                child.set_status(Status(StatusCode.ERROR, marker))
                with raw_tracer.start_as_current_span("grandchild") as grandchild:
                    grandchild_context = grandchild.get_span_context()
                    self.assertEqual(grandchild_context.trace_id, child_context.trace_id)

        exported = {span.name: span for span in exporter.get_finished_spans()}
        self.assertEqual(set(exported), {"RAGService.answer", "instrumented-child", "grandchild"})
        root = exported["RAGService.answer"]
        child_span = exported["instrumented-child"]
        grandchild_span = exported["grandchild"]
        self.assertEqual(child_span.parent.span_id, root.context.span_id)
        self.assertEqual(grandchild_span.parent.span_id, child_span.context.span_id)
        self.assertEqual({span.context.trace_id for span in exported.values()}, {root.context.trace_id})
        for span in exported.values():
            self.assertNotIn(marker, str(span.attributes))
            self.assertNotIn(marker, str(span.events))
            self.assertNotIn(marker, str(span.links))
            self.assertNotIn(marker, str(span.status.description))

    def test_rag_parent_span_does_not_export_exception_details_when_installed(self) -> None:
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
        except ImportError:
            self.skipTest("optional OpenTelemetry tracing extra is not installed")

        marker = "synthetic-private-marker-4821"
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with (
            patch("app.observability.tracing._tracing_configured", True),
            patch("opentelemetry.trace.get_tracer", return_value=provider.get_tracer("privacy-test")),
            self.assertRaisesRegex(ValueError, marker),
            trace_rag_query(marker),
        ):
            raise ValueError(marker)

        parent = exporter.get_finished_spans()[0]
        self.assertEqual(parent.events, ())
        self.assertNotIn(marker, str(parent.attributes))
        self.assertNotIn(marker, str(parent.status.description))

    def test_llm_log_uses_hash_and_redacted_preview(self) -> None:
        logger = CaptureLogger()
        service = LLMService(openai=None, cache=None, settings=build_settings())
        req = ChatRequest(
            messages=[
                {
                    "role": "user",
                    "content": "Проверь email ivan@mail.ru, тел +7 (999) 123-45-67, карта 4111 1111 1111 1111",
                }
            ]
        )
        response = ChatResponse(
            content="Ответ для ivan@mail.ru по заказу HD-123456",
            model="gpt-5.4-mini",
            usage=Usage(prompt_tokens=11, completion_tokens=3, total_tokens=14),
            finish_reason="stop",
        )

        with patch("app.services.llm.logger", logger):
            service._log_llm_response(req=req, response=response, raw_prompt=req.messages[0].content, latency_ms=12.34)

        event, fields = logger.events[0]
        self.assertEqual(event, "llm_request_completed")
        self.assertEqual(fields["input_tokens"], 11)
        self.assertEqual(fields["output_tokens"], 3)
        self.assertEqual(fields["latency_ms"], 12.34)
        self.assertIn("prompt_hash", fields)
        self.assertIn("[EMAIL]", fields["prompt_preview"])
        self.assertIn("[PHONE_RU]", fields["prompt_preview"])
        self.assertIn("[CARD]", fields["prompt_preview"])
        self.assertIn("[EMAIL]", fields["answer_preview"])
        self.assertIn("[ORDER_ID]", fields["answer_preview"])
        self.assertNotIn("ivan@mail.ru", fields["prompt_preview"])
        self.assertNotIn("+7 (999) 123-45-67", fields["prompt_preview"])
        self.assertNotIn("4111 1111 1111 1111", fields["prompt_preview"])
        self.assertNotIn("ivan@mail.ru", fields["answer_preview"])
        self.assertNotIn("HD-123456", fields["answer_preview"])


if __name__ == "__main__":
    unittest.main()
