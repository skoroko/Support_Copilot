# Эксперимент chunking и re-ranking — ДЗ 5.4

Дата прогона: `2026-09-09T15:03:34.388847+00:00`.

## Условия эксперимента

Golden dataset содержит **28** вопросов и покрывает **10** документов. Для каждого вопроса вручную указаны 1–3 релевантных документа. Метрики считаются на уровне исходного документа, а не отдельного chunk.

Embedding: `offline-lexical-hash-v2`, размерность `1024`. Прогон выполнен полностью локально с детерминированным лексическим embedding (`--offline-smoke`); документы не передавались внешнему провайдеру.

## Сравнение стратегий

| Стратегия | Коллекция | Chunks | Chunks/док. | Средняя длина, токены | Hit Rate@5 | MRR@10 | Recall@10 | Retrieval, мс |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed-512-64-k10` | `docs_fixed` | 3664 | 366.4 | 323.0 | 0.929 | 0.747 | 0.762 | 30.77 |
| `recursive-512-64-k10` | `docs_recursive` | 3680 | 368.0 | 304.72 | 0.929 | 0.771 | 0.750 | 25.53 |
| `semantic-p95-buffer1-k10` | `docs_semantic` | 3933 | 393.3 | 273.54 | 0.929 | 0.717 | 0.750 | 28.12 |

По приоритету Hit Rate@5 → MRR@10 → Recall@10 → latency лучшей базовой стратегией стала **`recursive`**.

## Re-ranker

| Вариант | Hit Rate@5 | MRR@10 | Recall@10 | Retrieval, мс |
|---|---:|---:|---:|---:|
| `without-reranker` | 0.929 | 0.771 | 0.750 | 29.27 |
| `with-BAAI/bge-reranker-v2-m3` | 0.929 | 0.839 | 0.720 | 18014.63 |

BGE изменил MRR@10 на +0.068, Recall@10 на -0.030, а Hit Rate@5 на +0.000. Средняя latency выросла с 29.27 до 18014.63 мс. На CPU re-ranker улучшает порядок первых результатов, но слишком медленен для интерактивного пути; он остаётся подключённым как опциональный этап для серверного/GPU-прогона.

## Подбор параметров

| Стратегия | Коллекция | Chunks | Chunks/док. | Средняя длина, токены | Hit Rate@5 | MRR@10 | Recall@10 | Retrieval, мс |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `recursive-256-32-k10` | `docs_recursive_256_32` | 8188 | 818.8 | 135.79 | 0.857 | 0.725 | 0.744 | 54.38 |
| `recursive-512-64-k10` | `docs_recursive` | 3680 | 368.0 | 304.72 | 0.929 | 0.771 | 0.750 | 24.14 |
| `recursive-512-64-k20` | `docs_recursive` | 3680 | 368.0 | 304.72 | 0.929 | 0.771 | 0.750 | 29.67 |
| `recursive-768-96-k20` | `docs_recursive_768_96` | 2730 | 273.0 | 409.74 | 0.929 | 0.766 | 0.750 | 22.03 |

## Итоговый выбор

Выбираю **`recursive`** с `chunk_size=512`, `overlap=64`, `top-K=10`. Эта конфигурация дала Hit Rate@5=0.929, MRR@10=0.771, Recall@10=0.750 и среднюю retrieval latency 24.14 мс.

При одинаковых quality-метриках у top-K=10 и top-K=20 выбран top-K=10: он передаёт в следующий этап вдвое меньше кандидатов, а разница локальной latency находится в пределах шума единичного прогона.

Целевой порог Hit Rate@5 ≥ 0.5 выполнен.

Параметры зафиксированы в `app/core/config.py` и `.env.example`. JSON с полными числами находится в `homework_5_4/results/chunking_experiment.json`.

## Ограничения

Офлайн lexical-hash embedding нужен для воспроизводимой проверки pipeline и не заменяет финальный прогон с выбранной embedding-моделью. Числа latency включают локальный in-memory Qdrant и не являются production benchmark.

## Референсы

- [LlamaIndex Node Parsers](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
- [LlamaIndex Retrieval Evaluation](https://docs.llamaindex.ai/en/stable/examples/evaluation/retrieval/retriever_eval/)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Qdrant Collections](https://qdrant.tech/documentation/concepts/collections/)
- [Pinecone Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [Cohere Reranking](https://docs.cohere.com/docs/reranking)

## Воспроизведение

```powershell
python scripts/evaluate_chunking.py --offline-smoke --skip-reranker
python scripts/evaluate_chunking.py --offline-smoke --reranker-local-files-only
```

Второй запуск выполняет локальный BGE re-ranking. Для облачного embedding требуется явное разрешение на передачу фрагментов документации и флаг `--allow-external-data-transfer`.
