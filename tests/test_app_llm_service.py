from __future__ import annotations

import unittest
from typing import Any

from pydantic import SecretStr

from app.core.config import Settings
from app.core.exceptions import LLMOutputBlockedError
from app.schemas.chat import ChatRequest
from app.services.llm import LLMService


class FakeCache:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.last_ttl: int | None = None

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def setex(self, key: str, ttl_seconds: int, value: str) -> bool:
        self.last_ttl = ttl_seconds
        self.values[key] = value
        return True


class FakeStream:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> "FakeStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class FakeCompletions:
    def __init__(self, stream_chunks: list[dict[str, Any]] | None = None) -> None:
        self.calls = 0
        self.last_kwargs: dict[str, Any] | None = None
        self.stream_chunks = stream_chunks

    async def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            return FakeStream(
                list(self.stream_chunks) if self.stream_chunks is not None else [
                    {"choices": [{"delta": {"content": "Первый "}}]},
                    {"choices": [{"delta": {"content": "чанк"}}]},
                    {"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
                ]
            )

        return {
            "model": kwargs["model"],
            "choices": [{"message": {"content": "Ответ из fake OpenAI"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


class FakeOpenAI:
    def __init__(self, stream_chunks: list[dict[str, Any]] | None = None) -> None:
        completions = FakeCompletions(stream_chunks)
        self.chat = type("Chat", (), {"completions": completions})()
        self.completions = completions


def build_settings() -> Settings:
    return Settings(
        openai_api_key=SecretStr("sk-test"),
        default_model="gpt-5.4-mini",
        request_timeout=30,
        redis_url="redis://localhost:6379/0",
        cache_ttl_seconds=123,
        cors_origins=["*"],
    )


class LLMServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_cache_key_includes_trusted_system_prompt(self) -> None:
        openai = FakeOpenAI()
        cache = FakeCache()
        service = LLMService(openai=openai, cache=cache, settings=build_settings())
        req = ChatRequest(messages=[{"role": "user", "content": "Как проверить лицензию?"}])

        first = await service.complete(req, system_prompt="trusted-a")
        second = await service.complete(req, system_prompt="trusted-a")
        third = await service.complete(req, system_prompt="trusted-b")

        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(first.content, "Ответ из fake OpenAI")
        self.assertEqual(second.content, first.content)
        self.assertFalse(third.cached)
        self.assertEqual(openai.completions.calls, 2)
        self.assertEqual(cache.last_ttl, 123)
        self.assertEqual(openai.completions.last_kwargs["max_completion_tokens"], 500)
        self.assertNotIn("max_tokens", openai.completions.last_kwargs)

    async def test_stream_yields_text_chunks_and_final_usage(self) -> None:
        openai = FakeOpenAI()
        service = LLMService(openai=openai, cache=FakeCache(), settings=build_settings())
        req = ChatRequest(messages=[{"role": "user", "content": "Стриминг"}])

        chunks = [delta async for delta in service.stream(req)]

        self.assertEqual([chunk.content for chunk in chunks if chunk.content is not None], ["Первый чанк"])
        self.assertEqual(chunks[-1].usage.total_tokens, 10)
        self.assertTrue(openai.completions.last_kwargs["stream"])
        self.assertEqual(openai.completions.last_kwargs["stream_options"], {"include_usage": True})
        self.assertEqual(openai.completions.last_kwargs["max_completion_tokens"], 500)

    async def test_stream_blocks_canary_split_across_provider_chunks_before_emitting(self) -> None:
        openai = FakeOpenAI(
            [
                {"choices": [{"delta": {"content": "CANARY_"}}]},
                {"choices": [{"delta": {"content": "deadbeef"}}]},
            ]
        )
        service = LLMService(openai=openai, cache=FakeCache(), settings=build_settings())
        req = ChatRequest(messages=[{"role": "user", "content": "Проверка"}])

        with self.assertRaises(LLMOutputBlockedError):
            _ = [delta async for delta in service.stream(req, canary="CANARY_deadbeef")]

    async def test_stream_redacts_email_split_across_provider_chunks(self) -> None:
        openai = FakeOpenAI(
            [
                {"choices": [{"delta": {"content": "Пишите alice@"}}]},
                {"choices": [{"delta": {"content": "example.com"}}]},
            ]
        )
        service = LLMService(openai=openai, cache=FakeCache(), settings=build_settings())
        req = ChatRequest(messages=[{"role": "user", "content": "Проверка"}])

        chunks = [delta async for delta in service.stream(req)]

        self.assertEqual([chunk.content for chunk in chunks], ["Пишите [EMAIL]"])


if __name__ == "__main__":
    unittest.main()
