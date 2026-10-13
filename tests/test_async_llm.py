from __future__ import annotations

import inspect
import time
import unittest

from support_copilot.config import SupportCopilotSettings
from support_copilot.infrastructure.async_llm import (
    AsyncLLMClient,
    build_demo_async_client,
)


class AsyncLLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_returns_demo_answer(self) -> None:
        client = build_demo_async_client(concurrency=2, delay_sec=0.001)

        answer = await client.complete("Как проверить лицензию?")

        self.assertIn("Support Copilot", answer)
        self.assertIn("demo", client.usage_summary())

    async def test_batch_chat_returns_exception_in_failed_position(self) -> None:
        client = build_demo_async_client(concurrency=2, delay_sec=0.001)

        results = await client.batch_chat(
            ["Первый запрос", "FORCE_ASYNC_ERROR", "Третий запрос"],
            concurrency=2,
        )

        self.assertIsInstance(results[0], str)
        self.assertIsInstance(results[1], Exception)
        self.assertIsInstance(results[2], str)

    async def test_batch_chat_rejects_mismatched_concurrency(self) -> None:
        client = build_demo_async_client(concurrency=2, delay_sec=0.001)

        with self.assertRaisesRegex(ValueError, "instance semaphore"):
            await client.batch_chat(["Первый запрос"], concurrency=5)

    async def test_stream_chat_yields_before_full_answer_finishes(self) -> None:
        client = build_demo_async_client(concurrency=1, delay_sec=0.002)
        start = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []

        async for delta in client.stream_chat("Что такое event loop?"):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            chunks.append(delta)

        total_time = time.perf_counter() - start
        self.assertIsNotNone(first_token_at)
        self.assertGreater(len(chunks), 1)
        self.assertLess(first_token_at - start, total_time)
        self.assertIn("Support Copilot", "".join(chunks))

    async def test_batch_overlaps_provider_calls_without_exceeding_concurrency(self) -> None:
        prompts = [f"Промпт {index}" for index in range(10)]
        for concurrency in (1, 5):
            client = build_demo_async_client(concurrency=concurrency, delay_sec=0)
            completions = client.providers[0].client.chat.completions
            original_create = completions.create
            active = peak = 0

            async def tracked_create(_create=original_create, **params):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                try:
                    # The demo provider yields to the event loop even at zero
                    # delay, so overlap can be measured without wall-clock noise.
                    return await _create(**params)
                finally:
                    active -= 1

            completions.create = tracked_create
            results = await client.batch_chat(prompts, concurrency=concurrency)
            self.assertEqual(len(results), len(prompts))
            self.assertTrue(all(isinstance(result, str) for result in results))
            self.assertEqual(peak, concurrency)
            self.assertEqual(active, 0)

    def test_semaphore_is_instance_attribute_not_method_local(self) -> None:
        client = build_demo_async_client(concurrency=3, delay_sec=0)

        self.assertTrue(hasattr(client, "_sem"))
        self.assertNotIn("Semaphore", inspect.getsource(AsyncLLMClient.complete))
        self.assertNotIn("Semaphore", inspect.getsource(AsyncLLMClient.batch_chat))

    def test_settings_has_async_defaults(self) -> None:
        settings = SupportCopilotSettings()

        self.assertEqual(settings.async_llm_concurrency, 5)
        self.assertEqual(settings.async_llm_sdk_timeout_sec, 30.0)
        self.assertEqual(settings.async_llm_business_timeout_sec, 15.0)
        self.assertEqual(settings.async_llm_max_retries, 3)


if __name__ == "__main__":
    unittest.main()
