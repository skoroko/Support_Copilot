from __future__ import annotations

from types import SimpleNamespace

import pytest
from qdrant_client.models import Distance, PointStruct

from app.services.vector_store import PAYLOAD_INDEXES, VectorStore, VectorStoreConfigurationError


class FakeQdrantClient:
    def __init__(self, *, exists: bool = False, size: int = 1024) -> None:
        self.exists = exists
        self.size = size
        self.payload_schema: dict[str, object] = {}
        self.created: list[dict[str, object]] = []
        self.indexes: list[dict[str, object]] = []
        self.upserts: list[dict[str, object]] = []
        self.query_response = SimpleNamespace(points=[SimpleNamespace(id="hit-1", score=0.9)])
        self.queries: list[dict[str, object]] = []
        self.closed = False

    async def collection_exists(self, _name: str) -> bool:
        return self.exists

    async def create_collection(self, **kwargs: object) -> bool:
        self.created.append(kwargs)
        self.exists = True
        self.size = kwargs["vectors_config"].size  # type: ignore[index,union-attr]
        return True

    async def get_collection(self, _name: str) -> object:
        vectors = SimpleNamespace(size=self.size, distance=Distance.COSINE)
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)),
            payload_schema=self.payload_schema,
        )

    async def create_payload_index(self, **kwargs: object) -> object:
        self.indexes.append(kwargs)
        self.payload_schema[str(kwargs["field_name"])] = object()
        return object()

    async def upsert(self, **kwargs: object) -> object:
        self.upserts.append(kwargs)
        return object()

    async def query_points(self, **kwargs: object) -> object:
        self.queries.append(kwargs)
        return self.query_response

    async def close(self) -> None:
        self.closed = True


def make_store(client: FakeQdrantClient, dimensions: int = 1024) -> VectorStore:
    return VectorStore(
        url="http://qdrant.test:6333",
        api_key="secret",
        collection_name="documents",
        embedding_dim=dimensions,
        client=client,
    )


@pytest.mark.asyncio
async def test_ensure_collection_creates_cosine_hnsw_and_payload_indexes() -> None:
    client = FakeQdrantClient()
    store = make_store(client)

    await store.ensure_collection()

    vectors = client.created[0]["vectors_config"]
    hnsw = client.created[0]["hnsw_config"]
    assert vectors.size == 1024
    assert vectors.distance == Distance.COSINE
    assert hnsw.m == 16
    assert hnsw.ef_construct == 100
    assert {call["field_name"] for call in client.indexes} == set(PAYLOAD_INDEXES)


@pytest.mark.asyncio
async def test_ensure_collection_rejects_dimension_mismatch() -> None:
    store = make_store(FakeQdrantClient(exists=True, size=768), dimensions=1024)

    with pytest.raises(VectorStoreConfigurationError, match="vector size 768"):
        await store.ensure_collection()


@pytest.mark.asyncio
async def test_upsert_batches_and_waits_only_for_last_batch() -> None:
    client = FakeQdrantClient(exists=True)
    store = make_store(client)
    points = [PointStruct(id=index, vector=[0.0] * 1024, payload={}) for index in range(513)]

    await store.upsert(points, batch_size=256)

    assert [len(call["points"]) for call in client.upserts] == [256, 256, 1]
    assert [call["wait"] for call in client.upserts] == [False, False, True]


@pytest.mark.asyncio
async def test_search_uses_query_points_and_returns_points() -> None:
    client = FakeQdrantClient(exists=True)
    store = make_store(client)

    hits = await store.search([0.0] * 1024, top_k=3)

    assert [hit.id for hit in hits] == ["hit-1"]
    assert client.queries[0]["collection_name"] == "documents"
    assert client.queries[0]["limit"] == 3
    assert client.queries[0]["with_payload"] is True
