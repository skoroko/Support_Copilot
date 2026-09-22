from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from support_copilot.api import app
from support_copilot.infrastructure.async_llm import build_demo_async_client


class SupportCopilotAPITests(unittest.TestCase):
    def test_health_and_sse_stream_with_demo_client(self) -> None:
        app.state.async_llm_client = build_demo_async_client(concurrency=2, delay_sec=0.001)

        with TestClient(
            app,
            headers={
                "X-Internal-Token": "test-internal-token-0000000000000001",
                "X-User-ID": "support-engineer-1",
            },
        ) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json(), {"status": "ok"})

            with client.stream(
                "POST",
                "/chat/stream",
                json={"prompt": "Что такое event loop?"},
            ) as response:
                self.assertEqual(response.status_code, 200)
                body = "".join(response.iter_text())

        self.assertIn("event: meta", body)
        self.assertIn("event: delta", body)
        self.assertIn("event: done", body)
        self.assertIn("Support", body)
        self.assertIn("Copilot", body)


if __name__ == "__main__":
    unittest.main()
