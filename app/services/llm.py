from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from openai import APITimeoutError, AuthenticationError, OpenAIError, RateLimitError

from app.core.config import Settings
from app.core.exceptions import LLMAuthError, LLMError, LLMRateLimitError, LLMTimeoutError
from app.observability.logging import get_logger
from app.observability.pii import prompt_hash, redact_pii
from app.schemas.chat import ChatDelta, ChatRequest, ChatResponse, Usage


logger = get_logger("llm-service")


def _cache_value_to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class LLMService:
    def __init__(self, openai: Any, cache: Any, settings: Settings) -> None:
        self.openai = openai
        self.cache = cache
        self.settings = settings

    async def complete(self, req: ChatRequest) -> ChatResponse:
        cache_key = self._cache_key(req)
        raw_prompt = self._prompt_text(req)
        cached = await self._cache_get(cache_key)
        if cached is not None:
            response = ChatResponse.model_validate_json(cached)
            cached_response = response.model_copy(update={"cached": True})
            self._log_llm_response(
                req=req,
                response=cached_response,
                raw_prompt=raw_prompt,
                latency_ms=0.0,
            )
            return cached_response

        model = req.model or self.settings.llm.default_model
        started = time.perf_counter()
        try:
            raw_response = await self.openai.chat.completions.create(
                model=model,
                messages=[message.model_dump() for message in req.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
        except RateLimitError as exc:
            raise LLMRateLimitError("LLM provider rate limit exceeded") from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError("LLM provider timeout") from exc
        except AuthenticationError as exc:
            raise LLMAuthError("LLM provider authentication failed") from exc
        except OpenAIError as exc:
            raise LLMError("LLM provider request failed") from exc

        latency_ms = (time.perf_counter() - started) * 1000
        response = ChatResponse.from_openai(raw_response, requested_model=model, cached=False)
        await self._cache_set(cache_key, response.model_dump_json())
        self._log_llm_response(req=req, response=response, raw_prompt=raw_prompt, latency_ms=latency_ms)
        return response

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        model = req.model or self.settings.llm.default_model
        try:
            stream = await self.openai.chat.completions.create(
                model=model,
                messages=[message.model_dump() for message in req.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                raw_usage = chunk.get("usage") if isinstance(chunk, dict) else getattr(chunk, "usage", None)
                usage = Usage.from_openai(raw_usage)
                if usage.total_tokens:
                    yield ChatDelta(usage=usage)
                    continue

                choices = chunk.get("choices", []) if isinstance(chunk, dict) else getattr(chunk, "choices", [])
                choices = choices or []
                if not choices:
                    continue
                first_choice = choices[0]
                delta = first_choice.get("delta") if isinstance(first_choice, dict) else getattr(first_choice, "delta", None)
                content = delta.get("content", "") if isinstance(delta, dict) else getattr(delta, "content", "")
                if content:
                    yield ChatDelta(content=str(content))

        except RateLimitError as exc:
            raise LLMRateLimitError("LLM provider rate limit exceeded") from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError("LLM provider timeout") from exc
        except AuthenticationError as exc:
            raise LLMAuthError("LLM provider authentication failed") from exc
        except OpenAIError as exc:
            raise LLMError("LLM provider request failed") from exc

    def _cache_key(self, req: ChatRequest) -> str:
        payload = req.model_dump(
            mode="json",
            exclude={"user_id", "session_id", "stream"},
            exclude_none=True,
        )
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "chat:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _prompt_text(self, req: ChatRequest) -> str:
        return "\n".join(message.content for message in req.messages if message.role in {"system", "user"})

    def _log_llm_response(
        self,
        *,
        req: ChatRequest,
        response: ChatResponse,
        raw_prompt: str,
        latency_ms: float,
    ) -> None:
        redacted_prompt = redact_pii(raw_prompt)
        logger.info(
            "llm_request_completed",
            model=response.model or req.model or self.settings.llm.default_model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=round(latency_ms, 2),
            finish_reason=response.finish_reason,
            prompt_hash=prompt_hash(raw_prompt),
            prompt_preview=redacted_prompt[:120],
            cached=response.cached,
        )

    async def _cache_get(self, key: str) -> str | None:
        try:
            return _cache_value_to_text(await self.cache.get(key))
        except Exception as exc:  # noqa: BLE001 - cache should not break LLM path
            logger.warning("cache_get_failed", key=key, error_type=type(exc).__name__)
            return None

    async def _cache_set(self, key: str, value: str) -> None:
        try:
            await self.cache.setex(key, self.settings.cache_ttl_seconds, value)
        except Exception as exc:  # noqa: BLE001 - cache should not break LLM path
            logger.warning("cache_set_failed", key=key, error_type=type(exc).__name__)
