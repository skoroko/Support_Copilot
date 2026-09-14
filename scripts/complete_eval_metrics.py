from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.eval.metrics import (  # noqa: E402
    METRIC_NAMES,
    aggregate_metrics,
    build_judge,
    build_metrics,
    eval_row,
)


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


async def run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(args.input)
    rows = frame.to_dict(orient="records")
    settings = get_settings()
    judge_model = args.judge_model or os.environ.get(
        "RAG_EVAL_JUDGE_MODEL", "gpt-4o-mini"
    )
    embedding_model = args.embedding_model or os.environ.get(
        "RAG_EVAL_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    models = build_judge(
        judge_model,
        embedding_model,
        api_key=os.environ.get("OPENAI_API_KEY")
        or settings.openai_api_key.get_secret_value(),
    )
    metrics = build_metrics(models)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{timestamp}_{args.label}.csv"
    completed_cells = 0
    completed_rows = 0
    checkpoint_lock = asyncio.Lock()
    case_semaphore = asyncio.Semaphore(args.concurrency)
    context_precision_concurrency = int(
        os.environ.get("RAG_EVAL_CONTEXT_PRECISION_CONCURRENCY", "2")
    )
    if context_precision_concurrency < 1:
        raise ValueError("RAG_EVAL_CONTEXT_PRECISION_CONCURRENCY must be positive")
    metric_semaphores = {
        "context_precision": asyncio.Semaphore(context_precision_concurrency)
    }

    async def complete_row(index: int, row: dict[str, Any]) -> None:
        nonlocal completed_cells, completed_rows
        async with case_semaphore:
            cached_values = {
                name: float(row[name])
                for name in METRIC_NAMES
                if name not in args.force_metric and _finite(row.get(name))
            }
            missing = [name for name in METRIC_NAMES if name not in cached_values]
            local_completed = 0
            if missing:
                raw_contexts = row.get("retrieved_contexts", "[]")
                contexts = (
                    json.loads(raw_contexts) if isinstance(raw_contexts, str) else []
                )
                values = await eval_row(
                    metrics,
                    user_input=str(row.get("user_input") or ""),
                    response=str(row.get("response") or ""),
                    retrieved_contexts=[str(item) for item in contexts],
                    reference=str(row.get("reference") or ""),
                    cached_values=cached_values,
                    metric_semaphores=metric_semaphores,
                )
                for name, value in values.items():
                    row[name] = value
                    if name in missing and name in METRIC_NAMES and _finite(value):
                        local_completed += 1
                        row.pop(f"{name}_error", None)
            async with checkpoint_lock:
                completed_cells += local_completed
                completed_rows += 1
                pd.DataFrame(rows).to_csv(
                    csv_path,
                    index=False,
                    encoding="utf-8-sig",
                )
                print(
                    f"completed {completed_rows}/{len(rows)} (row {index}); "
                    f"evaluated={','.join(missing) or 'none'}",
                    flush=True,
                )

    try:
        await asyncio.gather(
            *(complete_row(index, row) for index, row in enumerate(rows, start=1))
        )
    finally:
        await models.client.close()

    summary_source = args.metadata_json or args.input.with_suffix(".json")
    if summary_source.exists():
        summary = json.loads(summary_source.read_text(encoding="utf-8"))
    else:
        summary = {}
    configuration = dict(summary.get("configuration") or {})
    configuration.update(
        context_precision_transport="one_structured_call_per_case",
        context_precision_formula="ragas_average_precision",
        context_precision_concurrency=context_precision_concurrency,
        has_citation_method="deterministic_discrete_metric",
    )
    summary.update(
        generated_at=datetime.now(timezone.utc).isoformat(),
        label=args.label,
        case_count=len(rows),
        configuration=configuration,
        source_csv=str(args.input),
        forced_metrics=list(args.force_metric),
        completed_metric_cells=completed_cells,
        missing_metric_cells=sum(
            not _finite(row.get(name)) for row in rows for name in METRIC_NAMES
        ),
        metrics=aggregate_metrics(rows),
        csv=str(csv_path),
    )
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete missing RAGAS cells from an existing exact-input CSV"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tests/eval/results")
    parser.add_argument("--judge-model")
    parser.add_argument("--embedding-model")
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="Optional prior run JSON whose configuration is copied into the result",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("RAG_EVAL_CONCURRENCY", "3")),
    )
    parser.add_argument(
        "--force-metric",
        action="append",
        choices=METRIC_NAMES,
        default=[],
        help="Recompute this metric even when the input CSV already contains a value",
    )
    parser.add_argument("--allow-external-data-transfer", action="store_true")
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
