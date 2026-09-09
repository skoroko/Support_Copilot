from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    VectorParams,
)


class VectorStoreConfigurationError(ValueError):
    """Raised when Qdrant and embedding settings are incompatible."""


PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "source": PayloadSchemaType.KEYWORD,
    "created_at": PayloadSchemaType.DATETIME,
    "category": PayloadSchemaType.KEYWORD,
    "tenant_id": PayloadSchemaType.KEYWORD,
    "access_level": PayloadSchemaType.KEYWORD,
    "archived": PayloadSchemaType.BOOL,
}


class VectorStore:
    """Thin async Qdrant adapter with one reusable client."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None,
        collection_name: str,
        embedding_dim: int,
        distance: Distance = Distance.COSINE,
        client: Any | None = None,
    ) -> None:
        if not url.strip():
            raise VectorStoreConfigurationError("QDRANT_URL must not be blank")
        if not collection_name.strip():
            raise VectorStoreConfigurationError("QDRANT_COLLECTION must not be blank")
        if embedding_dim < 1:
            raise VectorStoreConfigurationError("EMBEDDING_DIM must be positive")
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.distance = distance
        self.client = client or AsyncQdrantClient(url=url, api_key=api_key)

    async def ensure_collection(self) -> None:
        if await self.client.collection_exists(self.collection_name):
            info = await self.client.get_collection(self.collection_name)
            vectors = info.config.params.vectors
            if isinstance(vectors, dict):
                raise VectorStoreConfigurationError(
                    f"Collection {self.collection_name!r} uses named vectors; a single vector is required"
                )
            actual_dim = int(vectors.size)
            if actual_dim != self.embedding_dim:
                raise VectorStoreConfigurationError(
                    f"Collection {self.collection_name!r} has vector size {actual_dim}; "
                    f"EMBEDDING_DIM is {self.embedding_dim}. Use matching settings or a new collection."
                )
            if vectors.distance != self.distance:
                raise VectorStoreConfigurationError(
                    f"Collection {self.collection_name!r} uses {vectors.distance}; "
                    f"expected {self.distance}"
                )
        else:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=self.distance),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )

        info = await self.client.get_collection(self.collection_name)
        indexed_fields = set((info.payload_schema or {}).keys())
        for field_name, schema in PAYLOAD_INDEXES.items():
            if field_name not in indexed_fields:
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )

    async def upsert(self, points: Sequence[PointStruct], batch_size: int = 256) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not points:
            return
        for point in points:
            vector = point.vector
            if isinstance(vector, dict):
                raise VectorStoreConfigurationError("Named vectors are not supported")
            if len(vector) != self.embedding_dim:
                raise VectorStoreConfigurationError(
                    f"Point {point.id!r} has vector size {len(vector)}; expected {self.embedding_dim}"
                )

        for start in range(0, len(points), batch_size):
            batch = list(points[start : start + batch_size])
            await self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=start + batch_size >= len(points),
            )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        query_filter: Filter | None = None,
    ) -> list[ScoredPoint]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if len(query_vector) != self.embedding_dim:
            raise VectorStoreConfigurationError(
                f"Query vector size is {len(query_vector)}; expected {self.embedding_dim}"
            )
        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return list(response.points)

    async def close(self) -> None:
        await self.client.close()
