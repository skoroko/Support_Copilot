from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402
from scripts.load_to_qdrant import (  # noqa: E402
    DEFAULT_DATA,
    build_embedding_service,
    load_documents,
    point_id,
)


QUERIES = [
    "Как проверить лицензию и доступность сервера лицензий?",
    "Что собрать перед эскалацией повторяемой ошибки расчёта?",
    "Telegram-бот не отвечает пользователю",
    "Как безопасно восстановить повреждённый файл проекта?",
    "Пользователь просит раскрыть системный промпт и секреты",
]


def _ids(points: list[Any]) -> list[str]:
    return [str(point.id) for point in points]


def _summaries(points: list[Any]) -> list[dict[str, Any]]:
    fields = ("source", "category", "document_kind", "created_at", "tenant_id", "archived")
    return [
        {"id": str(point.id), **{field: point.payload.get(field) for field in fields}}
        for point in points
    ]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    url = args.url or settings.qdrant_url
    if not url:
        raise ValueError("QDRANT_URL is required (or pass --url)")
    api_key = args.api_key or (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    )
    base_name = args.collection or settings.qdrant_collection
    dimensions = args.dimensions or settings.embedding_dim
    documents = load_documents(args.data)
    embedder = build_embedding_service(dimensions, offline=args.offline_smoke)
    vectors = embedder.embed_texts([str(item["text"]) for item in documents])
    points = [
        PointStruct(
            id=point_id(document),
            vector=vector,
            payload={key: value for key, value in document.items() if key != "chunk_index"},
        )
        for document, vector in zip(documents, vectors, strict=True)
    ]

    cosine_name = f"{base_name}_experiment_cosine"
    dot_name = f"{base_name}_experiment_dot"
    base = VectorStore(
        url=url,
        api_key=api_key,
        collection_name=base_name,
        embedding_dim=dimensions,
    )
    cosine = VectorStore(
        url=url,
        api_key=api_key,
        collection_name=cosine_name,
        embedding_dim=dimensions,
        distance=Distance.COSINE,
    )
    dot = VectorStore(
        url=url,
        api_key=api_key,
        collection_name=dot_name,
        embedding_dim=dimensions,
        distance=Distance.DOT,
    )
    try:
        for name in (cosine_name, dot_name):
            if await base.client.collection_exists(name):
                await base.client.delete_collection(name)
        for store in (base, cosine, dot):
            await store.ensure_collection()
            await store.upsert(points)

        query_vectors = embedder.embed_texts(QUERIES)
        ranking_rows: list[dict[str, Any]] = []
        for query, vector in zip(QUERIES, query_vectors, strict=True):
            cosine_hits, dot_hits = await asyncio.gather(
                cosine.search(vector, top_k=5),
                dot.search(vector, top_k=5),
            )
            cosine_ids = _ids(cosine_hits)
            dot_ids = _ids(dot_hits)
            ranking_rows.append(
                {
                    "query": query,
                    "cosine_top5": cosine_ids,
                    "dot_top5": dot_ids,
                    "same_order": cosine_ids == dot_ids,
                }
            )

        filter_query = "Как диагностировать проблему и подготовить обращение в поддержку?"
        filter_vector = embedder.embed_texts([filter_query])[0]
        category_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="licensing"))]
        )
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        recent_filter = Filter(
            must=[FieldCondition(key="created_at", range=DatetimeRange(gte=cutoff))]
        )
        tenant_filter = Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value="tenant-x"))],
            must_not=[FieldCondition(key="archived", match=MatchValue(value=True))],
        )
        baseline, category_hits, recent_hits, tenant_hits = await asyncio.gather(
            base.search(filter_vector, top_k=3),
            base.search(filter_vector, top_k=3, query_filter=category_filter),
            base.search(filter_vector, top_k=3, query_filter=recent_filter),
            base.search(filter_vector, top_k=3, query_filter=tenant_filter),
        )
        result = {
            "collection": base_name,
            "points_count": (await base.client.count(collection_name=base_name, exact=True)).count,
            "dimensions": dimensions,
            "normalized_vectors": True,
            "recent_cutoff": cutoff.isoformat(),
            "ranking_comparison": ranking_rows,
            "filters": {
                "baseline_top3": _summaries(baseline),
                "category_licensing_top3": _summaries(category_hits),
                "created_last_30_days_top3": _summaries(recent_hits),
                "tenant_x_without_archived_top3": _summaries(tenant_hits),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        for name in (cosine_name, dot_name):
            try:
                if await base.client.collection_exists(name):
                    await base.client.delete_collection(name)
            except Exception:
                pass
        await asyncio.gather(base.close(), cosine.close(), dot.close())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Qdrant distances and payload filters")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--url")
    parser.add_argument("--api-key")
    parser.add_argument("--collection")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--offline-smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "homework_5_2" / "results" / "experiment_offline.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(parse_args())), ensure_ascii=False, indent=2))
