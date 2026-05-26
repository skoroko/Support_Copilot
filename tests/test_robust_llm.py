from __future__ import annotations

import unittest
from typing import Any

from support_copilot.infrastructure import ProviderConfig, RobustLLMClient
from support_copilot.infrastructure.llm import _completion_tokens_parameter


class TestStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class RobustLLMTests(unittest.TestCase):
    def test_retries_then_fallback_provider(self) -> None:
        calls: list[str] = []

        def transport(
            provider: ProviderConfig,
            messages: list[dict[str, Any]],
            temperature: float,
            max_tokens: int,
        ) -> dict[str, Any]:
            calls.append(provider.name)
            if provider.name == "openai":
                raise TestStatusError(429)
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

        providers = [
            ProviderConfig("openai", "key", "https://api.openai.com/v1", "model"),
            ProviderConfig("openrouter", "key", "https://openrouter.ai/api/v1", "model"),
        ]
        client = RobustLLMClient(providers, sleep=lambda _: None, transport=transport)

        result = client.chat([{"role": "user", "content": "test"}])

        self.assertEqual(result.provider, "openrouter")
        self.assertTrue(result.fallback_used)
        self.assertEqual(calls.count("openai"), 5)
        self.assertEqual(calls.count("openrouter"), 1)

    def test_does_not_retry_400(self) -> None:
        calls = 0

        def transport(
            provider: ProviderConfig,
            messages: list[dict[str, Any]],
            temperature: float,
            max_tokens: int,
        ) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            raise TestStatusError(400)

        providers = [ProviderConfig("openai", "key", "https://api.openai.com/v1", "model")]
        client = RobustLLMClient(providers, sleep=lambda _: None, transport=transport)

        result = client.chat([{"role": "user", "content": "test"}])

        self.assertTrue(result.unavailable)
        self.assertEqual(calls, 1)

    def test_gpt5_uses_max_completion_tokens_parameter(self) -> None:
        self.assertEqual(
            _completion_tokens_parameter("gpt-5.4-mini"),
            "max_completion_tokens",
        )
        self.assertEqual(
            _completion_tokens_parameter("gpt-4.1-mini"),
            "max_tokens",
        )


if __name__ == "__main__":
    unittest.main()
