from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_TTL_SECONDS = 3600
DEFAULT_PROMPT_VERSION = "support-copilot-prompt-v1"
DEFAULT_SERVICE_FACTS_VERSION = "service-facts-v1"


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    writes: int
    entries: int
    expired: int
    hit_rate_percent: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "entries": self.entries,
            "expired": self.expired,
            "hit_rate_percent": round(self.hit_rate_percent, 2),
        }


@dataclass(frozen=True)
class CacheRecord:
    value: Any
    created_at: float
    expires_at: float


def canonical_payload(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    prompt_version: str,
    service_facts_version: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "prompt_version": prompt_version,
        "service_facts_version": service_facts_version,
    }


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cache_key(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    service_facts_version: str = DEFAULT_SERVICE_FACTS_VERSION,
) -> str:
    payload = canonical_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        prompt_version=prompt_version,
        service_facts_version=service_facts_version,
    )
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        *,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        service_facts_version: str = DEFAULT_SERVICE_FACTS_VERSION,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        self.ttl_seconds = ttl_seconds
        self.prompt_version = prompt_version
        self.service_facts_version = service_facts_version
        self.clock = clock
        self._records: dict[str, CacheRecord] = {}
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._expired = 0
        self._lock = threading.Lock()

    def make_key(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        *,
        prompt_version: str | None = None,
        service_facts_version: str | None = None,
    ) -> str:
        return cache_key(
            model=model,
            messages=messages,
            temperature=temperature,
            prompt_version=prompt_version or self.prompt_version,
            service_facts_version=service_facts_version or self.service_facts_version,
        )

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
        now = self.clock()

        with self._lock:
            record = self._records.get(key)
            if record is None:
                self._misses += 1
                return None
            if record.expires_at <= now:
                self._records.pop(key, None)
                self._misses += 1
                self._expired += 1
                return None
            self._hits += 1
            return record.value

    def set(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        value: Any,
        *,
        ttl_seconds: int | None = None,
        prompt_version: str | None = None,
        service_facts_version: str | None = None,
    ) -> str:
        ttl = ttl_seconds or self.ttl_seconds
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive.")

        key = self.make_key(
            model,
            messages,
            temperature,
            prompt_version=prompt_version,
            service_facts_version=service_facts_version,
        )
        now = self.clock()
        record = CacheRecord(value=value, created_at=now, expires_at=now + ttl)

        with self._lock:
            self._records[key] = record
            self._writes += 1
        return key

    def prune_expired(self) -> int:
        now = self.clock()
        with self._lock:
            expired_keys = [
                key for key, record in self._records.items() if record.expires_at <= now
            ]
            for key in expired_keys:
                self._records.pop(key, None)
            self._expired += len(expired_keys)
            return len(expired_keys)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            total_reads = self._hits + self._misses
            hit_rate = 0.0 if total_reads == 0 else self._hits / total_reads * 100
            snapshot = CacheStats(
                hits=self._hits,
                misses=self._misses,
                writes=self._writes,
                entries=len(self._records),
                expired=self._expired,
                hit_rate_percent=hit_rate,
            )
        return snapshot.as_dict()


class RedisLLMCache:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        *,
        namespace: str = "support-copilot:llm-cache",
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        service_facts_version: str = DEFAULT_SERVICE_FACTS_VERSION,
        redis_client: Any | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")

        self.ttl_seconds = ttl_seconds
        self.namespace = namespace
        self.prompt_version = prompt_version
        self.service_facts_version = service_facts_version
        self.clock = clock
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._expired = 0
        self._lock = threading.Lock()

        if redis_client is not None:
            self.client = redis_client
        else:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "Install Redis support first: python -m pip install redis"
                ) from exc
            self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    def make_key(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        *,
        prompt_version: str | None = None,
        service_facts_version: str | None = None,
    ) -> str:
        digest = cache_key(
            model=model,
            messages=messages,
            temperature=temperature,
            prompt_version=prompt_version or self.prompt_version,
            service_facts_version=service_facts_version or self.service_facts_version,
        )
        return f"{self.namespace}:{digest}"

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
        raw_value = self.client.get(key)
        with self._lock:
            if raw_value is None:
                self._misses += 1
                return None
            self._hits += 1

        payload = json.loads(raw_value)
        return payload["value"]

    def set(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        value: Any,
        *,
        ttl_seconds: int | None = None,
        prompt_version: str | None = None,
        service_facts_version: str | None = None,
    ) -> str:
        ttl = ttl_seconds or self.ttl_seconds
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive.")

        key = self.make_key(
            model,
            messages,
            temperature,
            prompt_version=prompt_version,
            service_facts_version=service_facts_version,
        )
        payload = {
            "value": value,
            "created_at": self.clock(),
            "ttl_seconds": ttl,
        }
        self.client.setex(key, ttl, stable_json(payload))
        with self._lock:
            self._writes += 1
        return key

    def clear(self) -> None:
        pattern = f"{self.namespace}:*"
        keys = list(self.client.scan_iter(match=pattern))
        if keys:
            self.client.delete(*keys)

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            total_reads = self._hits + self._misses
            hit_rate = 0.0 if total_reads == 0 else self._hits / total_reads * 100
            entries = len(list(self.client.scan_iter(match=f"{self.namespace}:*")))
            snapshot = CacheStats(
                hits=self._hits,
                misses=self._misses,
                writes=self._writes,
                entries=entries,
                expired=self._expired,
                hit_rate_percent=hit_rate,
            )
        return snapshot.as_dict()
