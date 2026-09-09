from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from collections.abc import Callable
from typing import Any

import tiktoken
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from tenacity import Retrying, before_sleep_log, retry_if_exception, stop_after_attempt, wait_random_exponential


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "var" / "embeddings.sqlite3"


class EmbeddingConfigurationError(ValueError):
    """Raised when embedding settings are invalid or incomplete."""


class EmbeddingResponseError(RuntimeError):
    """Raised when a provider returns malformed embeddings."""


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "openai"
    model: str = "text-embedding-3-large"
    dimensions: int | None = 1024
    batch_size: int = 100
    cache_path: Path = DEFAULT_CACHE_PATH
    request_timeout_seconds: float = 30.0
    max_attempts: int = 3
    retry_min_seconds: float = 0.25
    retry_max_seconds: float = 4.0
    max_input_tokens: int = 8192
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None

    def __post_init__(self) -> None:
        if self.provider != "openai":
            raise EmbeddingConfigurationError("Only EMBEDDING_PROVIDER=openai is supported in homework 5.1")
        if not self.model.strip():
            raise EmbeddingConfigurationError("EMBEDDING_MODEL must not be blank")
        if self.dimensions is not None and self.dimensions < 1:
            raise EmbeddingConfigurationError("EMBEDDING_DIMENSIONS must be positive or blank")
        if not 32 <= self.batch_size <= 2048:
            raise EmbeddingConfigurationError("EMBEDDING_BATCH_SIZE must be between 32 and 2048")
        if self.max_attempts < 1:
            raise EmbeddingConfigurationError("EMBEDDING_MAX_ATTEMPTS must be at least 1")
        if self.request_timeout_seconds <= 0:
            raise EmbeddingConfigurationError("EMBEDDING_TIMEOUT_SECONDS must be positive")
        if self.retry_min_seconds < 0 or self.retry_max_seconds < self.retry_min_seconds:
            raise EmbeddingConfigurationError("Embedding retry delays are invalid")
        if self.max_input_tokens < 1:
            raise EmbeddingConfigurationError("EMBEDDING_MAX_INPUT_TOKENS must be positive")

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        raw_dimensions = os.getenv("EMBEDDING_DIMENSIONS", "1024").strip()
        raw_cache_path = Path(os.getenv("EMBEDDING_CACHE_PATH", str(DEFAULT_CACHE_PATH)))
        if not raw_cache_path.is_absolute():
            raw_cache_path = PROJECT_ROOT / raw_cache_path

        return cls(
            provider=os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower(),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large").strip(),
            dimensions=int(raw_dimensions) if raw_dimensions else None,
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            cache_path=raw_cache_path,
            request_timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30")),
            max_attempts=int(os.getenv("EMBEDDING_MAX_ATTEMPTS", "3")),
            retry_min_seconds=float(os.getenv("EMBEDDING_RETRY_MIN_SECONDS", "0.25")),
            retry_max_seconds=float(os.getenv("EMBEDDING_RETRY_MAX_SECONDS", "4")),
            max_input_tokens=int(os.getenv("EMBEDDING_MAX_INPUT_TOKENS", "8192")),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )


@dataclass(frozen=True)
class EmbeddingRunStats:
    input_texts: int = 0
    unique_cache_misses: int = 0
    cache_hits: int = 0
    provider_requests: int = 0
    provider_texts: int = 0
    elapsed_seconds: float = 0.0


class SQLiteEmbeddingCache:
    """Small persistent cache keyed by provider, model, dimensions and text."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    vector_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        found: dict[str, list[float]] = {}
        unique_keys = list(dict.fromkeys(keys))
        with self._lock, self._connect() as connection:
            for start in range(0, len(unique_keys), 900):
                chunk = unique_keys[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT cache_key, vector_json FROM embeddings WHERE cache_key IN ({placeholders})",
                    chunk,
                ).fetchall()
                for cache_key, raw_vector in rows:
                    try:
                        vector = json.loads(raw_vector)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if isinstance(vector, list) and vector:
                        found[str(cache_key)] = [float(value) for value in vector]
        return found

    def set_many(self, values: dict[str, list[float]]) -> None:
        if not values:
            return
        now = int(time.time())
        rows = [
            (key, json.dumps(vector, separators=(",", ":")), now)
            for key, vector in values.items()
        ]
        with self._lock, self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO embeddings(cache_key, vector_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    vector_json = excluded.vector_json,
                    created_at = excluded.created_at
                """,
                rows,
            )


def _is_retryable_provider_error(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise EmbeddingResponseError("Provider returned a zero-length embedding vector")
    return [value / norm for value in vector]


class EmbeddingService:
    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        client: Any | None = None,
        cache: SQLiteEmbeddingCache | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig.from_env()
        if client is None:
            if not self.config.api_key:
                raise EmbeddingConfigurationError("OPENAI_API_KEY is required for the OpenAI embedding provider")
            client_kwargs: dict[str, Any] = {
                "api_key": self.config.api_key,
                "timeout": self.config.request_timeout_seconds,
            }
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            client = OpenAI(**client_kwargs)
        self._client = client
        self._cache = cache or SQLiteEmbeddingCache(self.config.cache_path)
        if token_counter is None:
            encoding = self._load_encoding(self.config.model)
            token_counter = lambda text: len(encoding.encode(text))
        self._token_counter = token_counter
        self.last_stats = EmbeddingRunStats()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        self._validate_texts(texts)
        if not texts:
            self.last_stats = EmbeddingRunStats(elapsed_seconds=time.perf_counter() - started)
            return []

        keys = [self._cache_key(text) for text in texts]
        cached = self._cache.get_many(keys)
        missing_by_key: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in cached:
                missing_by_key.setdefault(key, text)

        provider_requests = 0
        provider_texts = 0
        missing_items = list(missing_by_key.items())
        for start in range(0, len(missing_items), self.config.batch_size):
            batch_items = missing_items[start : start + self.config.batch_size]
            batch_texts = [text for _, text in batch_items]
            batch_vectors = self._request_batch(batch_texts)
            provider_requests += 1
            provider_texts += len(batch_texts)
            for (key, _), vector in zip(batch_items, batch_vectors, strict=True):
                cached[key] = vector

        if missing_by_key:
            self._cache.set_many({key: cached[key] for key in missing_by_key})

        result = [list(cached[key]) for key in keys]
        self.last_stats = EmbeddingRunStats(
            input_texts=len(texts),
            unique_cache_misses=len(missing_by_key),
            cache_hits=len(texts) - sum(1 for key in keys if key in missing_by_key),
            provider_requests=provider_requests,
            provider_texts=provider_texts,
            elapsed_seconds=time.perf_counter() - started,
        )
        return result

    def _request_batch(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.config.dimensions is not None:
            kwargs["dimensions"] = self.config.dimensions

        retryer = Retrying(
            stop=stop_after_attempt(self.config.max_attempts),
            wait=wait_random_exponential(
                multiplier=self.config.retry_min_seconds,
                max=self.config.retry_max_seconds,
            ),
            retry=retry_if_exception(_is_retryable_provider_error),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        response = retryer(self._client.embeddings.create, **kwargs)
        data = response["data"] if isinstance(response, dict) else response.data
        if len(data) != len(texts):
            raise EmbeddingResponseError(
                f"Provider returned {len(data)} vectors for {len(texts)} input texts"
            )

        ordered: list[tuple[int, list[float]]] = []
        for position, item in enumerate(data):
            index = item.get("index", position) if isinstance(item, dict) else getattr(item, "index", position)
            raw_vector = item["embedding"] if isinstance(item, dict) else item.embedding
            vector = [float(value) for value in raw_vector]
            if not vector:
                raise EmbeddingResponseError("Provider returned an empty embedding vector")
            if self.config.dimensions is not None and len(vector) != self.config.dimensions:
                raise EmbeddingResponseError(
                    f"Expected {self.config.dimensions} dimensions, received {len(vector)}"
                )
            ordered.append((int(index), _l2_normalize(vector)))

        ordered.sort(key=lambda pair: pair[0])
        if [index for index, _ in ordered] != list(range(len(texts))):
            raise EmbeddingResponseError("Provider response contains invalid embedding indexes")
        return [vector for _, vector in ordered]

    def _validate_texts(self, texts: list[str]) -> None:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list[str]")
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] must be str")
            if not text.strip():
                raise ValueError(f"texts[{index}] must not be blank")
            token_count = self._token_counter(text)
            if token_count > self.config.max_input_tokens:
                raise ValueError(
                    f"texts[{index}] has {token_count} tokens; limit is {self.config.max_input_tokens}"
                )

    def _cache_key(self, text: str) -> str:
        payload = {
            "provider": self.config.provider,
            "model": self.config.model,
            "dimensions": self.config.dimensions,
            "text": text,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "embedding:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_encoding(model: str) -> Any:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def reset_embedding_service() -> None:
    """Reload environment-based settings on the next module-level call."""

    get_embedding_service.cache_clear()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return normalized embeddings in input order, using batching and persistent cache."""

    return get_embedding_service().embed_texts(texts)
