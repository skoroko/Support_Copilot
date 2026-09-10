"""CLI entry point for safe, incremental RAG ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llama_index.core.embeddings import MockEmbedding
from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.ingestion import IngestionConfig, IngestionService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally ingest Support Copilot documents into Qdrant")
    parser.add_argument("paths", nargs="*", type=Path, help="Files or directories to ingest (default: --data-dir)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--api-key")
    parser.add_argument("--collection", default="support_copilot")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--docstore-path", type=Path, default=ROOT / "var" / "ingestion_docstore.json")
    parser.add_argument(
        "--allow-external-data-transfer",
        action="store_true",
        help="Confirm sending source fragments to configured hosted services",
    )
    parser.add_argument(
        "--offline-smoke",
        action="store_true",
        help="Use persistent local Qdrant and deterministic mock embeddings; sends nothing to external services",
    )
    parser.add_argument(
        "--offline-qdrant-path",
        type=Path,
        default=ROOT / "var" / "offline_qdrant",
        help="Persistent Qdrant directory for --offline-smoke",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    config = IngestionConfig(
        data_dir=args.data_dir,
        qdrant_url=args.url,
        qdrant_api_key=args.api_key,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.dimensions,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        docstore_path=args.docstore_path,
    )
    client = None
    embed_model = None
    if args.offline_smoke:
        args.offline_qdrant_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(args.offline_qdrant_path))
        embed_model = MockEmbedding(embed_dim=args.dimensions)
    service = IngestionService(config, client=client, embed_model=embed_model)
    try:
        result = service.ingest(
            args.paths or args.data_dir,
            allow_external_data_transfer=args.allow_external_data_transfer,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        print(result.summary)
        return 0 if not result.failed_files else 2
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
