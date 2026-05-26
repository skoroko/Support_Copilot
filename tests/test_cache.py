from __future__ import annotations

import unittest

from support_copilot.infrastructure.cache import LLMCache


class CacheTests(unittest.TestCase):
    def test_hit_miss_and_ttl(self) -> None:
        now = 1000.0

        def clock() -> float:
            return now

        cache = LLMCache(ttl_sec=10, clock=clock)
        messages = [{"role": "user", "content": "Как установить лицензию?"}]

        self.assertIsNone(cache.get("model", messages, 0.2))
        cache.set("model", messages, 0.2, "answer")
        self.assertEqual(cache.get("model", messages, 0.2), "answer")

        now = 1011.0
        self.assertIsNone(cache.get("model", messages, 0.2))
        self.assertEqual(cache.stats()["hits"], 1)
        self.assertEqual(cache.stats()["misses"], 2)

    def test_key_depends_on_temperature(self) -> None:
        cache = LLMCache()
        messages = [{"role": "user", "content": "test"}]
        first = cache.make_key("model", messages, 0.2)
        second = cache.make_key("model", messages, 0.7)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
