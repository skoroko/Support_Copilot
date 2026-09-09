from __future__ import annotations

import argparse
import json
from typing import Any


PRICING_CHECKED_ON = "2026-09-09"
DEFAULT_CANDIDATES = (
    ("text-embedding-3-small", 0.02, 1536),
    ("text-embedding-3-large", 0.13, 1024),
)


def estimate_indexing_cost(
    *,
    documents: int,
    pages_per_document: int,
    tokens_per_page: int,
    model: str,
    price_per_million_tokens: float,
    dimensions: int,
) -> dict[str, Any]:
    if min(documents, pages_per_document, tokens_per_page, dimensions) < 1:
        raise ValueError("Document, page, token and dimension counts must be positive")
    if price_per_million_tokens < 0:
        raise ValueError("Price must not be negative")

    chunks = documents * pages_per_document
    input_tokens = chunks * tokens_per_page
    synchronous_cost = input_tokens / 1_000_000 * price_per_million_tokens
    return {
        "model": model,
        "dimensions": dimensions,
        "documents": documents,
        "chunks": chunks,
        "input_tokens": input_tokens,
        "synchronous_cost_usd": round(synchronous_cost, 6),
        "batch_cost_usd": round(synchronous_cost * 0.5, 6),
        "float32_vector_storage_mib": round(chunks * dimensions * 4 / 1024**2, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate one-time embedding index cost")
    parser.add_argument("--documents", type=int, default=60)
    parser.add_argument("--pages-per-document", type=int, default=20)
    parser.add_argument("--tokens-per-page", type=int, default=800)
    args = parser.parse_args()

    payload = {
        "pricing_checked_on": PRICING_CHECKED_ON,
        "assumptions": {
            "documents": args.documents,
            "pages_per_document": args.pages_per_document,
            "tokens_per_page": args.tokens_per_page,
            "one_chunk_per_page": True,
        },
        "candidates": [
            estimate_indexing_cost(
                documents=args.documents,
                pages_per_document=args.pages_per_document,
                tokens_per_page=args.tokens_per_page,
                model=model,
                price_per_million_tokens=price,
                dimensions=dimensions,
            )
            for model, price, dimensions in DEFAULT_CANDIDATES
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
