from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.rag import RAGConfig, RAGService  # noqa: E402
from app.services.retrieval_eval import (  # noqa: E402
    answer_evidence_metrics,
    citation_provenance_metrics,
    load_retrieval_dataset,
    retrieval_metrics,
)


DATASET_PATH = ROOT / "tests" / "eval" / "retrieval_dataset.json"
RESULT_PATH = ROOT / "homework_5_5" / "results" / "rag_quality_evaluation.json"


def evaluate(service: RAGService, dataset: Path) -> dict[str, Any]:
    cases = load_retrieval_dataset(dataset)
    answers: list[str] = []
    source_rankings: list[list[dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    for case in cases:
        response = service.answer(case.question, history=case.history)
        sources = list(response["sources"])
        answer = str(response["answer"])
        answers.append(answer)
        source_rankings.append(sources)
        rows.append(
            {
                "question": case.question,
                "history": [asdict(item) for item in case.history],
                "answer": answer,
                "top_score": response["top_score"],
                "sources": sources,
            }
        )

    document_rankings = [
        [str(source.get("file_name") or "") for source in sources]
        for sources in source_rankings
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "case_count": len(cases),
        "metrics": {
            **retrieval_metrics(cases, document_rankings),
            **citation_provenance_metrics(cases, source_rankings),
            **answer_evidence_metrics(cases, answers),
        },
        "results": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the production RAG answer, citations and PDF page provenance"
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument(
        "--allow-external-data-transfer",
        action="store_true",
        help="Confirm that questions and selected manual fragments may be sent to the configured providers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_external_data_transfer:
        raise SystemExit(
            "Refusing to send manual fragments externally. Re-run with "
            "--allow-external-data-transfer only after explicit authorization."
        )
    config = replace(
        RAGConfig.from_settings(get_settings()),
        allow_external_data_transfer=True,
    )
    service = RAGService(config).build()
    try:
        payload = evaluate(service, args.dataset)
    finally:
        service.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
