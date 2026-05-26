"""LLM response caches."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class LLMCache:
    """In-memory LLM cache with TTL and hit/miss stats."""

    def __init__(
        self,
        ttl_sec: int = 3600,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_sec <= 0:
            raise ValueError("ttl_sec must be positive")
        self.ttl_sec = ttl_sec
        self._clock = clock or time.time
        self._store: dict[str, _CacheEntry] = {}
        self.hits = 0
        self.misses = 0

    def make_key(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        *,
        prompt_version: str | None = None,
        service_facts_version: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if prompt_version:
            payload["prompt_version"] = prompt_version
        if service_facts_version:
            payload["service_facts_version"] = service_facts_version

        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        *,
        prompt_version: str | None = None,
        service_facts_version: str | None = None,
    ) -> Any | None:
        key = self.make_key(
            model,
            messages,
            temperature,
            prompt_version=prompt_version,
            service_facts_version=service_facts_version,
        )
        entry = self._store.get(key)
        now = self._clock()
        if entry is None:
            self.misses += 1
            return None
        if entry.expires_at <= now:
            self._store.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        return entry.value

    def set(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        value: Any,
        *,
        prompt_version: str | None = None,
        service_facts_version: str | None = None,
    ) -> str:
        key = self.make_key(
            model,
            messages,
            temperature,
            prompt_version=prompt_version,
            service_facts_version=service_facts_version,
        )
        self._store[key] = _CacheEntry(value=value, expires_at=self._clock() + self.ttl_sec)
        return key

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict[str, float | int]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "size": len(self._store),
            "ttl_sec": self.ttl_sec,
        }


class RedisLLMCache:
    """Redis cache with the same public interface as LLMCache."""

    def __init__(self, redis_url: str, ttl_sec: int = 3600, namespace: str = "llm") -> None:
        if ttl_sec <= 0:
            raise ValueError("ttl_sec must be positive")
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Install redis-py to use RedisLLMCache") from exc

        self.ttl_sec = ttl_sec
        self.namespace = namespace
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.hits = 0
        self.misses = 0

    def make_key(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        *,
        prompt_version: str | None = None,
        service_facts_version: str | None = None,
    ) -> str:
        raw = LLMCache().make_key(
            model,
            messages,
            temperature,
            prompt_version=prompt_version,
            service_facts_version=service_facts_version,
        )
        return f"{self.namespace}:{raw}"

    def get(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        *,
        prompt_version: str | None = None,
        service_facts_version: str | None = None,
    ) -> Any | None:
        key = self.make_key(
            model,
            messages,
            temperature,
            prompt_version=prompt_version,
            service_facts_version=service_facts_version,
        )
        raw = self.client.get(key)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        value: Any,
        *,
        prompt_version: str | None = None,
        service_facts_version: str | None = None,
    ) -> str:
        key = self.make_key(
            model,
            messages,
            temperature,
            prompt_version=prompt_version,
            service_facts_version=service_facts_version,
        )
        self.client.setex(key, self.ttl_sec, json.dumps(value, ensure_ascii=False))
        return key

    def stats(self) -> dict[str, float | int]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "ttl_sec": self.ttl_sec,
        }
