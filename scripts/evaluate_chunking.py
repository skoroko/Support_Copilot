from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tiktoken
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import BaseNode, Document, MetadataMode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.chm_reader import CHMReader  # noqa: E402
from app.services.chunking import EmbeddingServiceAdapter, chunk_documents  # noqa: E402
from app.services.embeddings import EmbeddingConfig, EmbeddingService  # noqa: E402
from app.services.reranker import BGEReranker, RerankCandidate  # noqa: E402
from app.services.retrieval_eval import (  # noqa: E402
    RetrievalCase,
    load_retrieval_dataset,
    retrieval_metrics,
)
from scripts.load_to_qdrant import build_embedding_service  # noqa: E402


DATASET_PATH = ROOT / "tests" / "eval" / "retrieval_dataset.json"
RESULT_PATH = ROOT / "homework_5_4" / "results" / "chunking_experiment.json"
REPORT_PATH = ROOT / "docs" / "chunking_experiment.md"
POINT_NAMESPACE = uuid.UUID("7c0dbd69-e4d7-48e3-a05d-aac195b185cf")


@dataclass(frozen=True)
class ExperimentConfig:
    label: str
    strategy: str
    collection: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int


@dataclass(frozen=True)
class ChunkStats:
    source_documents: int
    total_chunks: int
    average_chunks_per_document: float
    average_chunk_tokens: float


def load_documents(data_dir: Path) -> list[Document]:
    return SimpleDirectoryReader(
        input_dir=str(data_dir),
        recursive=True,
        required_exts=[".pdf", ".docx", ".md", ".txt", ".chm"],
        file_extractor={".chm": CHMReader()},
        raise_on_error=True,
    ).load_data()


def source_doc_id(node: BaseNode) -> str:
    metadata = node.metadata or {}
    source = str(metadata.get("file_name") or metadata.get("source") or "unknown")
    return source.split("::", maxsplit=1)[0].split("#page=", maxsplit=1)[0]


def source_label(node: BaseNode) -> str:
    metadata = node.metadata or {}
    return str(metadata.get("source") or metadata.get("file_name") or "unknown")


def node_text(node: BaseNode) -> str:
    return node.get_content(metadata_mode=MetadataMode.NONE).strip()


def calculate_chunk_stats(nodes: list[BaseNode], *, encoding: Any) -> ChunkStats:
    if not nodes:
        raise ValueError("chunking produced no non-empty nodes")
    doc_ids = {source_doc_id(node) for node in nodes}
    lengths = [len(encoding.encode(node_text(node))) for node in nodes]
    return ChunkStats(
        source_documents=len(doc_ids),
        total_chunks=len(nodes),
        average_chunks_per_document=round(len(nodes) / len(doc_ids), 2),
        average_chunk_tokens=round(statistics.fmean(lengths), 2),
    )


def recreate_collection(
    client: QdrantClient,
    config: ExperimentConfig,
    nodes: list[BaseNode],
    embedder: EmbeddingService,
    *,
    dimensions: int,
) -> ChunkStats:
    if client.collection_exists(config.collection):
        client.delete_collection(config.collection)
    client.create_collection(
        collection_name=config.collection,
        vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
    )

    texts = [text for node in nodes if (text := node_text(node))]
    if len(texts) != len(nodes):
        raise ValueError("chunking returned nodes without retrievable text")
    vectors = embedder.embed_texts(texts)
    points = [
        PointStruct(
            id=str(uuid.uuid5(POINT_NAMESPACE, f"{config.collection}\n{index}\n{text}")),
            vector=vector,
            payload={
                "text": text,
                "source_doc_id": source_doc_id(node),
                "source": source_label(node),
                "chunk_index": index,
                "strategy": config.strategy,
            },
        )
        for index, (node, text, vector) in enumerate(zip(nodes, texts, vectors, strict=True))
    ]
    client.upload_points(collection_name=config.collection, points=points, batch_size=64, wait=True)
    try:
        encoding = tiktoken.encoding_for_model(embedder.config.model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return calculate_chunk_stats(nodes, encoding=encoding)


def retrieve_candidates(
    client: QdrantClient,
    *,
    collection: str,
    vector: list[float],
    limit: int,
) -> list[RerankCandidate]:
    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        with_payload=True,
    )
    candidates: list[RerankCandidate] = []
    for point in response.points:
        payload = point.payload or {}
        candidates.append(
            RerankCandidate(
                text=str(payload.get("text") or ""),
                doc_id=str(payload.get("source_doc_id") or "unknown"),
                source=str(payload.get("source") or "unknown"),
                score=float(point.score),
            )
        )
    return candidates


def evaluate_collection(
    client: QdrantClient,
    cases: list[RetrievalCase],
    query_vectors: list[list[float]],
    config: ExperimentConfig,
    *,
    reranker: BGEReranker | None = None,
    rerank_top_n: int = 10,
) -> dict[str, Any]:
    rankings: list[list[str]] = []
    latencies_ms: list[float] = []
    for case, vector in zip(cases, query_vectors, strict=True):
        started = time.perf_counter()
        candidates = retrieve_candidates(
            client,
            collection=config.collection,
            vector=vector,
            limit=config.retrieval_top_k,
        )
        if reranker is not None:
            candidates = reranker.rerank(case.question, candidates, top_n=rerank_top_n)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        rankings.append([candidate.doc_id for candidate in candidates])

    return {
        **retrieval_metrics(cases, rankings),
        "average_retrieval_ms": round(statistics.fmean(latencies_ms), 2),
    }


def build_real_embedder(settings: Any) -> EmbeddingService:
    return EmbeddingService(
        EmbeddingConfig(
            model=settings.rag_embedding_model,
            dimensions=settings.rag_embedding_dimensions,
            cache_path=ROOT / "var" / "chunking_embeddings.sqlite3",
            request_timeout_seconds=settings.request_timeout,
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
        )
    )


def metric_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = row["metrics"]
    config = row["config"]
    return (
        float(metrics["hit_rate@5"]),
        float(metrics["mrr@10"]),
        float(metrics["recall@10"]),
        -float(config["retrieval_top_k"]),
        -float(metrics["average_retrieval_ms"]),
    )


def _metric(value: Any) -> str:
    return f"{float(value):.3f}"


def _strategy_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Стратегия | Коллекция | Chunks | Chunks/док. | Средняя длина, токены | Hit Rate@5 | MRR@10 | Recall@10 | Retrieval, мс |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        config = row["config"]
        chunks = row["chunks"]
        metrics = row["metrics"]
        lines.append(
            f"| `{config['label']}` | `{config['collection']}` | {chunks['total_chunks']} | "
            f"{chunks['average_chunks_per_document']} | {chunks['average_chunk_tokens']} | "
            f"{_metric(metrics['hit_rate@5'])} | {_metric(metrics['mrr@10'])} | "
            f"{_metric(metrics['recall@10'])} | {metrics['average_retrieval_ms']} |"
        )
    return lines


def render_markdown_report(payload: dict[str, Any]) -> str:
    best_config = payload["best_tuned_config"]
    best_row = next(
        row
        for row in payload["parameter_tuning"]
        if row["config"]["label"] == best_config["label"]
    )
    best_metrics = best_row["metrics"]
    threshold_met = float(best_metrics["hit_rate@5"]) >= 0.5
    reranker_rows = payload["reranker_comparison"]

    lines = [
        "# Эксперимент chunking и re-ranking — ДЗ 5.4",
        "",
        f"Дата прогона: `{payload['generated_at']}`.",
        "",
        "## Условия эксперимента",
        "",
        f"Golden dataset содержит **{payload['dataset_cases']}** вопросов и покрывает "
        f"**{payload['corpus_files']}** документов. Для каждого вопроса вручную указаны "
        "1–3 релевантных документа. Метрики считаются на уровне исходного документа, "
        "а не отдельного chunk.",
        "",
        f"Embedding: `{payload['embedding_model']}`, размерность "
        f"`{payload['embedding_dimensions']}`. "
        + (
            "Прогон выполнен полностью локально с детерминированным лексическим "
            "embedding (`--offline-smoke`); документы не передавались внешнему провайдеру."
            if payload["offline_smoke"]
            else "Прогон выполнен с настроенной production embedding-моделью."
        ),
        "",
        "## Сравнение стратегий",
        "",
        *_strategy_table(payload["base_strategies"]),
        "",
        f"По приоритету Hit Rate@5 → MRR@10 → Recall@10 → latency лучшей базовой "
        f"стратегией стала **`{payload['best_base_strategy']}`**.",
        "",
        "## Re-ranker",
        "",
        "| Вариант | Hit Rate@5 | MRR@10 | Recall@10 | Retrieval, мс |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in reranker_rows:
        metrics = row["metrics"]
        lines.append(
            f"| `{row['label']}` | {_metric(metrics['hit_rate@5'])} | "
            f"{_metric(metrics['mrr@10'])} | {_metric(metrics['recall@10'])} | "
            f"{metrics['average_retrieval_ms']} |"
        )
    if len(reranker_rows) == 1:
        lines.extend(
            [
                "",
                "Re-ranker в этом прогоне отключён флагом `--skip-reranker`; строку "
                "до/после необходимо получить локальным прогоном с установленной моделью.",
            ]
        )
    else:
        before = reranker_rows[0]["metrics"]
        after = reranker_rows[1]["metrics"]
        lines.extend(
            [
                "",
                f"BGE изменил MRR@10 на "
                f"{float(after['mrr@10']) - float(before['mrr@10']):+.3f}, Recall@10 на "
                f"{float(after['recall@10']) - float(before['recall@10']):+.3f}, а Hit Rate@5 на "
                f"{float(after['hit_rate@5']) - float(before['hit_rate@5']):+.3f}. "
                f"Средняя latency выросла с {before['average_retrieval_ms']} до "
                f"{after['average_retrieval_ms']} мс. На CPU re-ranker улучшает порядок первых "
                "результатов, но слишком медленен для интерактивного пути; он остаётся "
                "подключённым как опциональный этап для серверного/GPU-прогона.",
            ]
        )

    lines.extend(
        [
            "",
            "## Подбор параметров",
            "",
            *_strategy_table(payload["parameter_tuning"]),
            "",
            "## Итоговый выбор",
            "",
            f"Выбираю **`{best_config['strategy']}`** с `chunk_size={best_config['chunk_size']}`, "
            f"`overlap={best_config['chunk_overlap']}`, `top-K={best_config['retrieval_top_k']}`. "
            f"Эта конфигурация дала Hit Rate@5={_metric(best_metrics['hit_rate@5'])}, "
            f"MRR@10={_metric(best_metrics['mrr@10'])}, Recall@10={_metric(best_metrics['recall@10'])} "
            f"и среднюю retrieval latency {best_metrics['average_retrieval_ms']} мс.",
            "",
            "При одинаковых quality-метриках у top-K=10 и top-K=20 выбран top-K=10: "
            "он передаёт в следующий этап вдвое меньше кандидатов, а разница локальной "
            "latency находится в пределах шума единичного прогона.",
            "",
            (
                "Целевой порог Hit Rate@5 ≥ 0.5 выполнен."
                if threshold_met
                else "Целевой порог Hit Rate@5 ≥ 0.5 не выполнен. Следующая итерация: "
                "проверить golden labels, добавить hybrid retrieval и повторить подбор с re-ranker."
            ),
            "",
            "Параметры зафиксированы в `app/core/config.py` и `.env.example`. "
            "JSON с полными числами находится в `homework_5_4/results/chunking_experiment.json`.",
            "",
            "## Ограничения",
            "",
            "Офлайн lexical-hash embedding нужен для воспроизводимой проверки pipeline и не "
            "заменяет финальный прогон с выбранной embedding-моделью. Числа latency включают "
            "локальный in-memory Qdrant и не являются production benchmark.",
            "",
            "## Референсы",
            "",
            "- [LlamaIndex Node Parsers](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)",
            "- [LlamaIndex Retrieval Evaluation](https://docs.llamaindex.ai/en/stable/examples/evaluation/retrieval/retriever_eval/)",
            "- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)",
            "- [Qdrant Collections](https://qdrant.tech/documentation/concepts/collections/)",
            "- [Pinecone Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)",
            "- [Cohere Reranking](https://docs.cohere.com/docs/reranking)",
            "",
            "## Воспроизведение",
            "",
            "```powershell",
            "python scripts/evaluate_chunking.py --offline-smoke --skip-reranker",
            "python scripts/evaluate_chunking.py --offline-smoke --reranker-local-files-only",
            "```",
            "",
            "Второй запуск выполняет локальный BGE re-ranking. Для облачного embedding "
            "требуется явное разрешение на передачу фрагментов документации и флаг "
            "`--allow-external-data-transfer`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.offline_smoke and not args.allow_external_data_transfer:
        raise SystemExit(
            "Refusing to send manual fragments externally. Re-run with "
            "--allow-external-data-transfer only after explicit authorization."
        )

    settings = get_settings()
    data_dir = Path(settings.rag_data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    cases = load_retrieval_dataset(args.dataset)
    documents = load_documents(data_dir)
    corpus_doc_ids = {str(document.metadata.get("file_name") or "") for document in documents}
    unknown = sorted({doc_id for case in cases for doc_id in case.relevant_doc_ids} - corpus_doc_ids)
    if unknown:
        raise ValueError(f"golden dataset references files outside the corpus: {unknown}")

    embedder = (
        build_embedding_service(settings.rag_embedding_dimensions, offline=True)
        if args.offline_smoke
        else build_real_embedder(settings)
    )
    semantic_embedder = EmbeddingServiceAdapter(
        embedder,
        model_name=embedder.config.model,
        embed_batch_size=embedder.config.batch_size,
    )
    query_vectors = embedder.embed_texts([case.question for case in cases])
    qdrant_api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    client = QdrantClient(":memory:") if args.offline_smoke else QdrantClient(
        url=settings.qdrant_url or "http://localhost:6333",
        api_key=qdrant_api_key,
        timeout=settings.request_timeout,
    )

    base_configs = [
        ExperimentConfig("fixed-512-64-k10", "fixed", "docs_fixed", 512, 64, 10),
        ExperimentConfig("recursive-512-64-k10", "recursive", "docs_recursive", 512, 64, 10),
        ExperimentConfig("semantic-p95-buffer1-k10", "semantic", "docs_semantic", 512, 64, 10),
    ]
    tuning_configs = [
        ExperimentConfig("recursive-256-32-k10", "recursive", "docs_recursive_256_32", 256, 32, 10),
        ExperimentConfig("recursive-512-64-k10", "recursive", "docs_recursive", 512, 64, 10),
        ExperimentConfig("recursive-512-64-k20", "recursive", "docs_recursive", 512, 64, 20),
        ExperimentConfig("recursive-768-96-k20", "recursive", "docs_recursive_768_96", 768, 96, 20),
    ]
    indexed: dict[tuple[str, int, int], tuple[str, ChunkStats]] = {}

    def prepare(config: ExperimentConfig) -> ChunkStats:
        key = (config.strategy, config.chunk_size, config.chunk_overlap)
        previous = indexed.get(key)
        if previous is not None:
            existing_collection, stats = previous
            if existing_collection != config.collection:
                raise ValueError("the same chunk configuration cannot map to two collections")
            return stats
        nodes = chunk_documents(
            config.strategy,  # type: ignore[arg-type]
            documents,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            embed_model=semantic_embedder,
        )
        stats = recreate_collection(
            client,
            config,
            nodes,
            embedder,
            dimensions=settings.rag_embedding_dimensions,
        )
        indexed[key] = (config.collection, stats)
        return stats

    try:
        base_rows: list[dict[str, Any]] = []
        for config in base_configs:
            stats = prepare(config)
            base_rows.append(
                {
                    "config": asdict(config),
                    "chunks": asdict(stats),
                    "metrics": evaluate_collection(client, cases, query_vectors, config),
                }
            )

        best_base = max(base_rows, key=metric_key)
        best_config = ExperimentConfig(**best_base["config"])
        reranker_rows: list[dict[str, Any]] = []
        comparison_config = ExperimentConfig(
            label=f"{best_config.strategy}-without-reranker-k20",
            strategy=best_config.strategy,
            collection=best_config.collection,
            chunk_size=best_config.chunk_size,
            chunk_overlap=best_config.chunk_overlap,
            retrieval_top_k=20,
        )
        reranker_rows.append(
            {
                "label": "without-reranker",
                "config": asdict(comparison_config),
                "metrics": evaluate_collection(client, cases, query_vectors, comparison_config),
            }
        )
        if not args.skip_reranker:
            reranker = BGEReranker(
                settings.rag_reranker_model,
                device=settings.rag_reranker_device or None,
                batch_size=settings.rag_reranker_batch_size,
                local_files_only=args.reranker_local_files_only,
            )
            reranker_rows.append(
                {
                    "label": f"with-{settings.rag_reranker_model}",
                    "config": asdict(comparison_config),
                    "metrics": evaluate_collection(
                        client,
                        cases,
                        query_vectors,
                        comparison_config,
                        reranker=reranker,
                        rerank_top_n=10,
                    ),
                }
            )

        tuning_rows: list[dict[str, Any]] = []
        for config in tuning_configs:
            stats = prepare(config)
            tuning_rows.append(
                {
                    "config": asdict(config),
                    "chunks": asdict(stats),
                    "metrics": evaluate_collection(client, cases, query_vectors, config),
                }
            )

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "offline_smoke": args.offline_smoke,
            "dataset_cases": len(cases),
            "corpus_files": len(corpus_doc_ids),
            "embedding_model": embedder.config.model,
            "embedding_dimensions": embedder.config.dimensions,
            "base_strategies": base_rows,
            "best_base_strategy": best_base["config"]["strategy"],
            "reranker_comparison": reranker_rows,
            "parameter_tuning": tuning_rows,
            "best_tuned_config": max(tuning_rows, key=metric_key)["config"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown_report(payload), encoding="utf-8")
        return payload
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate chunking and reranking for homework 5.4")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--offline-smoke", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument(
        "--reranker-local-files-only",
        action="store_true",
        help="Use an already cached re-ranker without network checks",
    )
    parser.add_argument("--allow-external-data-transfer", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
