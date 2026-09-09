from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import PointStruct
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.embeddings import EmbeddingConfig, EmbeddingService, SQLiteEmbeddingCache  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402


DEFAULT_DATA = ROOT / "data" / "support_documents.json"


class _OfflineEmbeddingsEndpoint:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def create(self, *, input: list[str], **_: Any) -> dict[str, Any]:
        rows = []
        for index, text in enumerate(input):
            vector = [0.0] * self.dimensions
            words = re.findall(r"[\w-]+", text.casefold(), flags=re.UNICODE)
            features: list[tuple[str, float]] = [(f"word:{word}", 2.0) for word in words]
            for word in words:
                padded = f"^{word}$"
                features.extend((f"tri:{padded[pos:pos + 3]}", 0.5) for pos in range(len(padded) - 2))
            for feature, weight in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                position = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[position] += sign * weight
            norm = math.sqrt(sum(value * value for value in vector))
            rows.append({"index": index, "embedding": [value / norm for value in vector]})
        return {"data": rows}


def build_embedding_service(dimensions: int, *, offline: bool) -> EmbeddingService:
    if not offline:
        service = EmbeddingService()
        if service.config.dimensions != dimensions:
            raise ValueError(
                f"EMBEDDING_DIM={dimensions} does not match EMBEDDING_DIMENSIONS="
                f"{service.config.dimensions} from homework 5.1"
            )
        return service

    config = EmbeddingConfig(
        model="offline-lexical-hash-v2",
        dimensions=dimensions,
        cache_path=ROOT / "var" / "offline_embeddings.sqlite3",
        api_key="offline",
    )
    client = SimpleNamespace(embeddings=_OfflineEmbeddingsEndpoint(dimensions))
    return EmbeddingService(
        config,
        client=client,
        cache=SQLiteEmbeddingCache(config.cache_path),
        token_counter=lambda text: len(text.split()),
    )


def load_documents(path: Path) -> list[dict[str, Any]]:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) < 100:
        raise ValueError(f"{path} must contain at least 100 documents")
    required = {"source", "chunk_index", "text", "created_at", "category"}
    for index, item in enumerate(raw):
        missing = required - set(item)
        if missing:
            raise ValueError(f"Document {index} is missing fields: {sorted(missing)}")
    return raw


def point_id(document: dict[str, Any]) -> str:
    return str(uuid5(NAMESPACE_URL, f"{document['source']}#{document['chunk_index']}"))


async def run(args: argparse.Namespace) -> int:
    if not 128 <= args.batch_size <= 256:
        raise ValueError("--batch-size must be between 128 and 256")
    settings = get_settings()
    url = args.url or settings.qdrant_url
    if not url:
        raise ValueError("QDRANT_URL is required (or pass --url)")
    api_key = args.api_key or (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    )
    collection = args.collection or settings.qdrant_collection
    dimensions = args.dimensions or settings.embedding_dim
    documents = load_documents(args.data)
    embedder = build_embedding_service(dimensions, offline=args.offline_smoke)

    texts = [str(document["text"]) for document in documents]
    with tqdm(total=len(texts), desc="Embedding", unit="doc") as progress:
        vectors = embedder.embed_texts(texts)
        progress.update(len(texts))
    points = [
        PointStruct(
            id=point_id(document),
            vector=vector,
            payload={key: value for key, value in document.items() if key != "chunk_index"},
        )
        for document, vector in zip(documents, vectors, strict=True)
    ]

    store = VectorStore(
        url=url,
        api_key=api_key,
        collection_name=collection,
        embedding_dim=dimensions,
    )
    try:
        await store.ensure_collection()
        await store.upsert(points, batch_size=args.batch_size)
        count = await store.client.count(collection_name=collection, exact=True)
        print(
            f"collection={collection} documents={len(documents)} points_count={count.count} "
            f"dimensions={dimensions} offline={args.offline_smoke}"
        )
        return int(count.count)
    finally:
        await store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Idempotently load Support Copilot documents into Qdrant")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--url")
    parser.add_argument("--api-key")
    parser.add_argument("--collection")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--offline-smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
