from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.eval.dataset import GoldenCase, load_golden_dataset  # noqa: E402
from app.eval.metrics import (  # noqa: E402
    METRIC_NAMES,
    aggregate_metrics,
    build_judge,
    build_metrics,
    eval_row,
)
from app.services.rag import RAGConfig, RAGService  # noqa: E402


DEFAULT_GOLDEN = ROOT / "tests" / "eval" / "golden_dataset.json"
DEFAULT_OUTPUT_DIR = ROOT / "tests" / "eval" / "results"

for noisy_logger in ("httpx", "openai", "ragas"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _build_rag(args: argparse.Namespace) -> RAGService:
    config = RAGConfig.from_settings(get_settings())
    config = replace(
        config,
        allow_external_data_transfer=True,
        chunk_size=args.chunk_size or config.chunk_size,
        chunk_overlap=args.chunk_overlap if args.chunk_overlap is not None else config.chunk_overlap,
        top_k=args.top_k or config.top_k,
        collection_name=args.collection or config.collection_name,
        ingest_on_start=args.ingest,
        docstore_path=args.docstore or config.docstore_path,
        rerank_enabled=args.reranker if args.reranker is not None else config.rerank_enabled,
    )
    return RAGService(config).build()


async def _evaluate_case(
    case: GoldenCase,
    *,
    rag: RAGService,
    metrics: dict[str, Any],
    metric_cache: dict[str, float],
    judge_model: str,
    embedding_model: str,
    semaphore: asyncio.Semaphore,
    metric_semaphores: dict[str, asyncio.Semaphore],
    collect_only: bool,
) -> dict[str, Any]:
    async with semaphore:
        started_at = time.perf_counter()
        rag_result = await asyncio.to_thread(rag.evaluate_inputs, case.user_input)
        latency_ms = (time.perf_counter() - started_at) * 1000
        retrieved_contexts = [
            str(item)
            for item in rag_result.get("retrieved_contexts", [])
            if str(item).strip()
        ]
        response = str(rag_result.get("answer") or "")
        metric_values: dict[str, Any] = {}
        if not collect_only:
            cached_values = {
                name: metric_cache[key]
                for name in METRIC_NAMES
                if (
                    key := _metric_cache_key(
                        name,
                        judge_model=judge_model,
                        embedding_model=embedding_model,
                        user_input=case.user_input,
                        response=response,
                        retrieved_contexts=retrieved_contexts,
                        reference=case.reference,
                    )
                )
                in metric_cache
            }
            metric_values = await eval_row(
                metrics,
                user_input=case.user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
                reference=case.reference,
                cached_values=cached_values,
                metric_semaphores=metric_semaphores,
            )
        return {
            **asdict(case),
            "response": response,
            "retrieved_contexts": retrieved_contexts,
            "sources": rag_result.get("sources", []),
            "top_score": rag_result.get("top_score", 0.0),
            "latency_ms": latency_ms,
            **metric_values,
        }


def _metric_cache_key(
    name: str,
    *,
    judge_model: str,
    embedding_model: str,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    reference: str,
) -> str:
    fields: dict[str, Any] = {
        "metric": name,
        "judge_model": judge_model,
        "embedding_model": embedding_model,
        "answer_relevancy_strictness": int(
            os.environ.get("RAG_EVAL_ANSWER_RELEVANCY_STRICTNESS", "1")
        ),
    }
    if name in {"faithfulness", "answer_relevancy"}:
        fields.update(user_input=user_input, response=response)
    if name in {"faithfulness", "context_precision", "context_recall"}:
        fields["retrieved_contexts"] = retrieved_contexts
    if name in {"context_precision", "context_recall"}:
        fields.update(user_input=user_input, reference=reference)
    if name == "has_citation":
        fields["response"] = response
    payload = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_metric_cache(
    paths: list[Path],
    *,
    judge_model: str,
    embedding_model: str,
) -> dict[str, float]:
    import math
    import pandas as pd

    cache: dict[str, float] = {}
    for path in paths:
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            raw_contexts = row.get("retrieved_contexts", "[]")
            contexts = json.loads(raw_contexts) if isinstance(raw_contexts, str) else []
            for name in METRIC_NAMES:
                value = row.get(name)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    continue
                key = _metric_cache_key(
                    name,
                    judge_model=judge_model,
                    embedding_model=embedding_model,
                    user_input=str(row.get("user_input") or ""),
                    response=str(row.get("response") or ""),
                    retrieved_contexts=[str(item) for item in contexts],
                    reference=str(row.get("reference") or ""),
                )
                cache[key] = float(value)
    return cache


async def run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    import pandas as pd

    cases = load_golden_dataset(args.golden)
    settings = get_settings()
    judge_model = args.judge_model or os.environ.get("RAG_EVAL_JUDGE_MODEL", "gpt-4o-mini")
    embedding_model = args.embedding_model or os.environ.get(
        "RAG_EVAL_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    models = None
    metrics: dict[str, Any] = {}
    metric_cache: dict[str, float] = {}
    if not args.collect_only:
        models = build_judge(
            judge_model,
            embedding_model,
            api_key=os.environ.get("OPENAI_API_KEY")
            or settings.openai_api_key.get_secret_value(),
        )
        metrics = build_metrics(models)
        metric_cache = _load_metric_cache(
            args.reuse_results,
            judge_model=judge_model,
            embedding_model=embedding_model,
        )
    rag = _build_rag(args)
    try:
        semaphore = asyncio.Semaphore(args.concurrency)
        context_precision_concurrency = int(
            os.environ.get("RAG_EVAL_CONTEXT_PRECISION_CONCURRENCY", "2")
        )
        if context_precision_concurrency < 1:
            raise ValueError(
                "RAG_EVAL_CONTEXT_PRECISION_CONCURRENCY must be positive"
            )
        metric_semaphores = {
            "context_precision": asyncio.Semaphore(context_precision_concurrency)
        }
        tasks = [
            asyncio.create_task(
                _evaluate_case(
                    case,
                    rag=rag,
                    metrics=metrics,
                    metric_cache=metric_cache,
                    judge_model=judge_model,
                    embedding_model=embedding_model,
                    semaphore=semaphore,
                    metric_semaphores=metric_semaphores,
                    collect_only=args.collect_only,
                )
            )
            for case in cases
        ]
        rows = []
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            rows.append(await task)
            print(f"evaluated {completed}/{len(tasks)}", flush=True)
    finally:
        rag.close()
        if models is not None:
            await models.client.close()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{timestamp}_{args.label}.csv"
    json_path = args.output_dir / f"{timestamp}_{args.label}.json"
    serializable_rows = []
    for row in rows:
        serializable = dict(row)
        for field in ("reference_contexts", "retrieved_contexts", "sources"):
            serializable[field] = json.dumps(serializable[field], ensure_ascii=False)
        serializable_rows.append(serializable)
    pd.DataFrame(serializable_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "case_count": len(rows),
        "configuration": {
            "judge_model": judge_model,
            "judge_embedding_model": embedding_model,
            "collect_only": args.collect_only,
            "judge_concurrency": args.concurrency,
            "judge_metric_timeout_seconds": float(
                os.environ.get("RAG_EVAL_METRIC_TIMEOUT_SECONDS", "180")
            ),
            "answer_relevancy_strictness": int(
                os.environ.get("RAG_EVAL_ANSWER_RELEVANCY_STRICTNESS", "1")
            ),
            "context_precision_transport": "one_structured_call_per_case",
            "context_precision_formula": "ragas_average_precision",
            "context_precision_concurrency": context_precision_concurrency,
            "has_citation_method": "deterministic_discrete_metric",
            "rag_llm_model": rag.config.llm_model,
            "rag_embedding_model": rag.config.embedding_model,
            "rag_embedding_dimensions": rag.config.embedding_dimensions,
            "chunk_size": rag.config.chunk_size,
            "chunk_overlap": rag.config.chunk_overlap,
            "top_k": rag.config.top_k,
            "reranker_enabled": rag.config.rerank_enabled,
            "reranker_model": rag.config.reranker_model if rag.config.rerank_enabled else None,
            "collection": rag.config.collection_name,
            "reused_result_files": [str(path) for path in args.reuse_results],
        },
        "metrics": aggregate_metrics(rows),
        "metric_cells_reused": sum(
            bool(row.get(f"{name}_cached")) for row in rows for name in METRIC_NAMES
        ),
        "csv": str(csv_path),
    }
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS 0.4 evaluation for Support Copilot")
    parser.add_argument("--label", required=True)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reuse-results",
        type=Path,
        action="append",
        default=[],
        help="Reuse metric cells only when their complete hashed inputs match a previous CSV",
    )
    parser.add_argument("--judge-model")
    parser.add_argument("--embedding-model")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("RAG_EVAL_CONCURRENCY", "3")),
    )
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--collection")
    parser.add_argument("--docstore", type=Path)
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Persist exact RAG answers and contexts without running judge metrics",
    )
    reranker = parser.add_mutually_exclusive_group()
    reranker.add_argument("--reranker", action="store_true", dest="reranker")
    reranker.add_argument("--no-reranker", action="store_false", dest="reranker")
    parser.set_defaults(reranker=None)
    parser.add_argument(
        "--allow-external-data-transfer",
        action="store_true",
        help="Required confirmation that questions and manual fragments may be sent to the judge",
    )
    args = parser.parse_args()
    if not args.allow_external_data_transfer:
        parser.error("--allow-external-data-transfer is required")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    return args


def main() -> None:
    csv_path, json_path, summary = asyncio.run(run(parse_args()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
