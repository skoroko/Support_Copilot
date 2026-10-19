"""Build an isolated S0-005 collection with production readers and metered APIs."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.eval.acceptance_budget import build_embedding, open_budget
from app.services.ingestion import IngestionConfig, IngestionService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--url", default="http://localhost:16333")
    parser.add_argument("--collection", default="s0005_candidate_v1")
    parser.add_argument("--limit-usd", type=float, default=10.)
    parser.add_argument("--allow-external-data-transfer", action="store_true")
    parser.add_argument("--visual-transcriptions", type=Path)
    parser.add_argument("--reviewed-only", action="store_true",
                        help="Reuse validated stored originals; update approved visual pages only")
    args = parser.parse_args()
    if not args.allow_external_data_transfer:
        parser.error("Explicit external data transfer confirmation is required")
    if args.limit_usd != 10.:
        parser.error("The authorized total task limit is exactly $10")
    target = urlsplit(args.url)
    if (target.scheme != "http" or target.hostname not in {"localhost", "127.0.0.1", "host.docker.internal"}
            or target.port != 16333 or target.path not in {"", "/"}
            or target.username or target.password or target.query or target.fragment):
        parser.error("Only the isolated local candidate Qdrant :16333 is permitted")
    settings = get_settings()
    endpoint = urlsplit(settings.openai_base_url or "https://api.openai.com/v1")
    if (endpoint.scheme != "https" or endpoint.hostname != "api.openai.com"
            or endpoint.port not in {None, 443} or endpoint.path.rstrip("/") != "/v1"
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        parser.error("Only the explicitly authorized official OpenAI endpoint is permitted")
    if not args.collection.startswith("s0005_candidate_") or args.collection == settings.rag_collection:
        parser.error("Only a distinct S0-005 candidate collection is permitted")
    # Log errors by type in the result; never dump SDK exception bodies.
    logging.disable(logging.CRITICAL)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    index = args.artifacts / "index"
    index.mkdir(exist_ok=True)
    ledger = open_budget(args.artifacts / "budget.sqlite3", limit_usd=args.limit_usd)
    config = replace(IngestionConfig.from_settings(settings),
        data_dir=args.data_dir.resolve(), qdrant_url=args.url, qdrant_api_key=None,
        collection_name=args.collection, docstore_path=index / "docstore.json",
        corpus_catalog_path=index / "catalog.json", visual_transcriptions_path=args.visual_transcriptions)
    embedding = build_embedding(ledger=ledger, model=config.embedding_model,
        dimensions=config.embedding_dimensions, api_key=config.openai_api_key,
        api_base=config.openai_base_url)
    service = IngestionService(config, embed_model=embedding, requires_external_data_transfer=True)
    attempt = {"collection": args.collection, "embedding_model": config.embedding_model,
               "dimensions": config.embedding_dimensions, "status": "started"}
    run_path = args.artifacts / "index" / "build_attempts.jsonl"
    with run_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(attempt) + "\n")
    try:
        result = (service.ingest_reviewed_transcriptions(allow_external_data_transfer=True)
                  if args.reviewed_only else service.ingest(allow_external_data_transfer=True))
        attempt.update(status="completed", result=result.as_dict())
        return_code = 0 if not result.failed_files and not result.unreadable_locations else 2
    except Exception as exc:  # noqa: BLE001 - only the safe exception class is persisted
        attempt.update(status="failed", error_type=type(exc).__name__)
        return_code = 1
    finally:
        service.close()
        attempt["budget"] = ledger.summary()
        with run_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(attempt, ensure_ascii=False) + "\n")
        print(json.dumps(attempt, ensure_ascii=False, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
