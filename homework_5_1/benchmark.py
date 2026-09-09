from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import tiktoken


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.embeddings import EmbeddingConfig, EmbeddingService  # noqa: E402


DEFAULT_BENCHMARK = PROJECT_ROOT / "tests" / "eval" / "mini_benchmark.json"
DEFAULT_CACHE = PROJECT_ROOT / "var" / "homework_5_1_embeddings.sqlite3"


class LocalHashEmbeddings:
    """Deterministic lexical vectors for an offline pipeline/cache smoke test."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        dimensions = int(kwargs.get("dimensions") or 256)
        data = [
            {"index": index, "embedding": self._vector(text, dimensions)}
            for index, text in enumerate(kwargs["input"])
        ]
        return {"data": data}

    @staticmethod
    def _vector(text: str, dimensions: int) -> list[float]:
        vector = [0.0] * dimensions
        words = re.findall(r"[\w-]+", text.lower(), flags=re.UNICODE)
        features = words + [word[:5] for word in words if len(word) > 5]
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            position = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[position] += sign
        return vector


class LocalHashClient:
    def __init__(self) -> None:
        self.embeddings = LocalHashEmbeddings()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        raise ValueError("Cannot compare a zero vector")
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def load_benchmark(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not 5 <= len(payload) <= 10:
        raise ValueError("Benchmark must contain 5-10 items")
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != {"query", "relevant", "irrelevant"}:
            raise ValueError(f"Benchmark item {index} has an invalid shape")
        if any(not isinstance(value, str) or not value.strip() for value in item.values()):
            raise ValueError(f"Benchmark item {index} contains blank text")
    return payload


def parse_candidate(raw: str) -> tuple[str, int | None]:
    model, separator, raw_dimensions = raw.partition(":")
    if not model.strip():
        raise argparse.ArgumentTypeError("Candidate model must not be blank")
    if not separator or not raw_dimensions.strip():
        return model.strip(), None
    try:
        dimensions = int(raw_dimensions)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Candidate dimensions must be an integer") from exc
    if dimensions < 1:
        raise argparse.ArgumentTypeError("Candidate dimensions must be positive")
    return model.strip(), dimensions


def _flatten(items: list[dict[str, str]]) -> list[str]:
    return [item[field] for item in items for field in ("query", "relevant", "irrelevant")]


def _token_count(texts: list[str], model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(text)) for text in texts)


def evaluate_candidate(
    candidate: tuple[str, int | None],
    items: list[dict[str, str]],
    *,
    cache_path: Path,
    client_factory: Callable[[], Any] | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    model, dimensions = candidate
    base_config = EmbeddingConfig.from_env()
    config = replace(
        base_config,
        model=model,
        dimensions=dimensions,
        cache_path=cache_path,
    )
    texts = _flatten(items)

    first_service = EmbeddingService(
        config,
        client=client_factory() if client_factory else None,
        token_counter=token_counter,
    )
    vectors = first_service.embed_texts(texts)
    first_stats = asdict(first_service.last_stats)

    restarted_service = EmbeddingService(
        config,
        client=client_factory() if client_factory else None,
        token_counter=token_counter,
    )
    repeated_vectors = restarted_service.embed_texts(texts)
    restart_stats = asdict(restarted_service.last_stats)
    if vectors != repeated_vectors:
        raise RuntimeError("Cached vectors changed after service restart")

    cases: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        query, relevant, irrelevant = vectors[index * 3 : index * 3 + 3]
        relevant_score = cosine_similarity(query, relevant)
        irrelevant_score = cosine_similarity(query, irrelevant)
        cases.append(
            {
                "query": item["query"],
                "relevant_score": round(relevant_score, 6),
                "irrelevant_score": round(irrelevant_score, 6),
                "margin": round(relevant_score - irrelevant_score, 6),
                "passed": relevant_score > irrelevant_score,
            }
        )

    return {
        "model": model,
        "dimensions": len(vectors[0]),
        "input_tokens": (
            sum(token_counter(text) for text in texts)
            if token_counter is not None
            else _token_count(texts, model)
        ),
        "accuracy": round(sum(case["passed"] for case in cases) / len(cases), 4),
        "mean_margin": round(sum(case["margin"] for case in cases) / len(cases), 6),
        "first_pass": first_stats,
        "after_restart": restart_stats,
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Support Copilot embedding mini-benchmark")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        default=None,
        help="MODEL:DIMENSIONS; repeat to compare candidates",
    )
    parser.add_argument(
        "--offline-smoke",
        action="store_true",
        help="Exercise batching, scoring and persistent cache without network or semantic-quality claims",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.offline_smoke:
        candidates = [("local-hash-smoke", 256)]
        client_factory: Callable[[], Any] | None = LocalHashClient
        token_counter: Callable[[str], int] | None = lambda text: len(text.split())
    else:
        candidates = args.candidate or [
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 1024),
        ]
        client_factory = None
        token_counter = None
    items = load_benchmark(args.benchmark)
    payload = {
        "mode": "offline-smoke" if args.offline_smoke else "openai",
        "benchmark": str(args.benchmark),
        "cases": len(items),
        "candidates": [
            evaluate_candidate(
                candidate,
                items,
                cache_path=args.cache_path,
                client_factory=client_factory,
                token_counter=token_counter,
            )
            for candidate in candidates
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
