from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import APITimeoutError, BadRequestError

from app.services.embeddings import EmbeddingConfig, EmbeddingService
from homework_5_1.benchmark import load_benchmark
from homework_5_1.cost_estimate import estimate_indexing_cost


ROOT = Path(__file__).resolve().parents[2]


def count_test_tokens(text: str) -> int:
    return len(text.split())


TEST_TOKEN_COUNTER = count_test_tokens


class FakeEmbeddings:
    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.failures = list(failures or [])

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        dimensions = kwargs.get("dimensions") or 3
        data = []
        for index, text in enumerate(kwargs["input"]):
            seed = float(sum(ord(character) for character in text))
            vector = [seed + index + offset + 1 for offset in range(dimensions)]
            data.append({"index": index, "embedding": vector})
        return {"data": data}


class FakeClient:
    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.embeddings = FakeEmbeddings(failures)


def config(tmp_path: Path, **overrides: Any) -> EmbeddingConfig:
    base = EmbeddingConfig(
        model="embedding-test",
        dimensions=3,
        batch_size=32,
        cache_path=tmp_path / "embeddings.sqlite3",
        max_attempts=1,
        retry_min_seconds=0,
        retry_max_seconds=0,
        api_key="test",
    )
    return replace(base, **overrides)


def test_mini_benchmark_has_required_shape_and_domain_examples() -> None:
    path = ROOT / "tests" / "eval" / "mini_benchmark.json"
    items = load_benchmark(path)

    assert len(items) == 8
    assert all(set(item) == {"query", "relevant", "irrelevant"} for item in items)
    joined = json.dumps(items, ensure_ascii=False).lower()
    for term in ("лицензи", "установ", "расчет", "документац", "prompt_injection"):
        assert term in joined


def test_embed_texts_batches_and_normalizes_vectors(tmp_path: Path) -> None:
    client = FakeClient()
    service = EmbeddingService(config(tmp_path), client=client, token_counter=TEST_TOKEN_COUNTER)

    vectors = service.embed_texts([f"support text {index}" for index in range(65)])

    assert len(client.embeddings.calls) == 3
    assert [len(call["input"]) for call in client.embeddings.calls] == [32, 32, 1]
    assert all(math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0) for vector in vectors)
    assert service.last_stats.provider_texts == 65


def test_persistent_cache_avoids_provider_after_restart(tmp_path: Path) -> None:
    settings = config(tmp_path)
    first_client = FakeClient()
    first = EmbeddingService(settings, client=first_client, token_counter=TEST_TOKEN_COUNTER)
    expected = first.embed_texts(["тот же текст"])

    restarted_client = FakeClient()
    restarted = EmbeddingService(settings, client=restarted_client, token_counter=TEST_TOKEN_COUNTER)
    actual = restarted.embed_texts(["тот же текст"])

    assert actual == expected
    assert len(first_client.embeddings.calls) == 1
    assert restarted_client.embeddings.calls == []
    assert restarted.last_stats.cache_hits == 1
    assert restarted.last_stats.provider_requests == 0


def test_model_change_invalidates_cache(tmp_path: Path) -> None:
    first_client = FakeClient()
    EmbeddingService(
        config(tmp_path, model="model-small"),
        client=first_client,
        token_counter=TEST_TOKEN_COUNTER,
    ).embed_texts(["один текст"])

    second_client = FakeClient()
    EmbeddingService(
        config(tmp_path, model="model-large"),
        client=second_client,
        token_counter=TEST_TOKEN_COUNTER,
    ).embed_texts(["один текст"])

    assert len(first_client.embeddings.calls) == 1
    assert len(second_client.embeddings.calls) == 1


def test_retry_handles_timeout_then_succeeds(tmp_path: Path) -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    client = FakeClient([APITimeoutError(request=request)])
    service = EmbeddingService(
        config(tmp_path, max_attempts=2),
        client=client,
        token_counter=TEST_TOKEN_COUNTER,
    )

    result = service.embed_texts(["retry timeout"])

    assert len(result) == 1
    assert len(client.embeddings.calls) == 2


def test_non_retryable_400_fails_once(tmp_path: Path) -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(400, request=request)
    client = FakeClient([BadRequestError("bad input", response=response, body=None)])
    service = EmbeddingService(
        config(tmp_path, max_attempts=3),
        client=client,
        token_counter=TEST_TOKEN_COUNTER,
    )

    with pytest.raises(BadRequestError):
        service.embed_texts(["invalid request"])

    assert len(client.embeddings.calls) == 1


def test_cache_preserves_duplicates_and_input_order(tmp_path: Path) -> None:
    service = EmbeddingService(
        config(tmp_path),
        client=FakeClient(),
        token_counter=TEST_TOKEN_COUNTER,
    )

    vectors = service.embed_texts(["alpha", "beta", "alpha"])

    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]
    assert service.last_stats.unique_cache_misses == 2


def test_rejects_blank_and_oversized_inputs(tmp_path: Path) -> None:
    service = EmbeddingService(
        config(tmp_path, max_input_tokens=2),
        client=FakeClient(),
        token_counter=TEST_TOKEN_COUNTER,
    )

    with pytest.raises(ValueError, match="must not be blank"):
        service.embed_texts([" "])
    with pytest.raises(ValueError, match="limit is 2"):
        service.embed_texts(["this input has too many tokens"])


def test_cost_estimate_for_sixty_document_corpus() -> None:
    estimate = estimate_indexing_cost(
        documents=60,
        pages_per_document=20,
        tokens_per_page=800,
        model="text-embedding-3-large",
        price_per_million_tokens=0.13,
        dimensions=1024,
    )

    assert estimate["chunks"] == 1200
    assert estimate["input_tokens"] == 960_000
    assert estimate["synchronous_cost_usd"] == 0.1248
    assert estimate["batch_cost_usd"] == 0.0624
    assert estimate["float32_vector_storage_mib"] == 4.6875
