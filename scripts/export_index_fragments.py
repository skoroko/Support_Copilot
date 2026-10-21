"""Read-only homework 5.2 adapter for real fragments and existing vectors.

No embedding client, ingestion, collection creation or provider call is used.
The text export passes load_to_qdrant.load_documents; its companion preserves
the existing vectors so a local experiment need not re-embed the fragments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def adapt_records(records: Iterable[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents, vectors = [], []
    dimensions = None
    seen = set()
    for record in records:
        payload = dict(record.payload or {})
        node = json.loads(payload["_node_content"]) if payload.get("_node_content") else {}
        metadata = {**payload, **node.get("metadata", {})}
        text = node.get("text") or payload.get("text")
        required = ("source", "created_at", "category")
        if not isinstance(text, str) or not text.strip() or any(not metadata.get(key) for key in required):
            raise ValueError("Index point lacks text/source/created_at/category; metadata must not be invented")
        vector = record.vector
        if not isinstance(vector, list) or not vector or any(not math.isfinite(float(value)) for value in vector):
            raise ValueError("A finite, unnamed dense vector is required")
        if dimensions is None:
            dimensions = len(vector)
        if dimensions != len(vector):
            raise ValueError("Mixed vector dimensions")
        identity = str(record.id)
        if identity in seen:
            raise ValueError("Duplicate point id")
        seen.add(identity)
        # Existing point identity is stable; no original chunk ordinal is invented.
        documents.append({
            **{key: metadata[key] for key in required},
            "chunk_index": identity, "text": text,
            "original_point_id": identity,
        })
        vectors.append({"original_point_id": identity, "vector": vector})
    if len(documents) < 100:
        raise ValueError("At least 100 real index fragments are required")
    return documents, vectors


def export(client: Any, *, collection: str, limit: int) -> tuple[list[dict], list[dict]]:
    if limit < 100:
        raise ValueError("limit must be at least 100")
    records = []
    offset = None
    while len(records) < limit:
        batch, offset = client.scroll(
            collection_name=collection, limit=min(100, limit - len(records)),
            offset=offset, with_payload=True, with_vectors=True,
        )
        records.extend(batch)
        if offset is None:
            break
    return adapt_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Existing local Qdrant URL")
    parser.add_argument("--collection", required=True, help="Existing collection; never modified")
    parser.add_argument("--env-file", type=Path, help="Existing configuration for QDRANT_API_KEY only")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    endpoint = urlsplit(args.url)
    if (endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
            or endpoint.scheme not in {"http", "https"}
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise SystemExit("Only a loopback Qdrant URL is permitted")
    from dotenv import dotenv_values
    from qdrant_client import QdrantClient

    values = dotenv_values(args.env_file) if args.env_file else {}
    client = QdrantClient(url=args.url, api_key=os.getenv("QDRANT_API_KEY") or values.get("QDRANT_API_KEY"))
    try:
        documents, vectors = export(client, collection=args.collection, limit=args.limit)
    finally:
        client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    companion = args.output.with_suffix(".vectors.json")
    companion.write_text(json.dumps({
        "exported_at": datetime.now(UTC).isoformat(),
        "source_collection": args.collection,
        "fragments_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "dimensions": len(vectors[0]["vector"]),
        "points": vectors,
        "scope": "Existing vectors; no new embeddings. Self-query tests do not measure semantic query quality.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(documents)} real fragments; existing collection unchanged.")
    print(args.output)
    print(companion)


if __name__ == "__main__":
    main()
