from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.rag import RAGConfig, RAGService  # noqa: E402
from app.services.rag_baremetal import BareMetalRAGService  # noqa: E402


QUESTIONS_PATH = ROOT / "homework_5_3" / "questions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Support Copilot RAG on five approved questions")
    parser.add_argument("--backend", choices=("llamaindex", "baremetal", "all"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "homework_5_3" / "results" / "rag_evaluation.json")
    parser.add_argument(
        "--allow-external-data-transfer",
        action="store_true",
        help="Confirm that manual fragments may be sent to the configured embedding and LLM provider",
    )
    return parser.parse_args()


def evaluate(name: str, service: Any, questions: list[dict[str, Any]]) -> dict[str, Any]:
    service.build()
    try:
        results = []
        for item in questions:
            response = service.answer(item["question"])
            results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": response["answer"],
                    "top_score": response["top_score"],
                    "top1_source": response["sources"][0]["source"] if response["sources"] else None,
                    "sources": response["sources"],
                    "expected_answer": item["expected_answer"],
                    "expected_sources": item["expected_sources"],
                    "hypothesis": item["hypothesis"],
                }
            )
        return {"backend": name, "results": results}
    finally:
        service.close()


def main() -> None:
    args = parse_args()
    if not args.allow_external_data_transfer:
        raise SystemExit(
            "Refusing to send manual fragments externally. Re-run with "
            "--allow-external-data-transfer only after explicit authorization."
        )
    settings = get_settings()
    config = RAGConfig.from_settings(settings)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    services = {
        "llamaindex": lambda: RAGService(config),
        "baremetal": lambda: BareMetalRAGService(config),
    }
    selected = services if args.backend == "all" else {args.backend: services[args.backend]}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "embedding_model": config.embedding_model,
            "embedding_dimensions": config.embedding_dimensions,
            "llm_model": config.llm_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "top_k": config.top_k,
        },
        "evaluations": [evaluate(name, factory(), questions) for name, factory in selected.items()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
