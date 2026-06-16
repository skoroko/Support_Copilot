from __future__ import annotations

import unittest
from unittest.mock import patch

from pydantic import SecretStr

from app.core.config import Settings
from app.observability.pii import prompt_hash, redact_pii
from app.observability.tracing import _normalize_phoenix_endpoint
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
        redacted = redact_pii("email@example.com, +7 999 123 45 67")

        self.assertNotIn("email@example.com", redacted)
        self.assertNotIn("+7 999 123 45 67", redacted)
        self.assertIn("[EMAIL]", redacted)
        self.assertIn("[PHONE_RU]", redacted)

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
        self.assertEqual(
            _normalize_phoenix_endpoint("http://phoenix:6006/v1/traces"),
            "http://phoenix:6006/v1/traces",
        )

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
            content="ok",
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
        self.assertNotIn("ivan@mail.ru", fields["prompt_preview"])
        self.assertNotIn("+7 (999) 123-45-67", fields["prompt_preview"])
        self.assertNotIn("4111 1111 1111 1111", fields["prompt_preview"])


if __name__ == "__main__":
    unittest.main()
