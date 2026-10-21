# Эксперимент chunking и re-ranking — ДЗ 5.4

Исторический offline эксперимент. Выбранные здесь 512/64 позднее заменены
на 1024/64 по [облачной оценке RAG от 12.09.2026](rag_evaluation.md).
Текущие параметры и границы доказательств: [итоговое описание](final_technical_documentation.md).

Дата прогона: `2026-09-11T08:53:00.406606+00:00`.

## Условия эксперимента

Golden dataset содержит **34** вопроса и покрывает **10** документов. Для содержательных вопросов вручную указаны 1–3 релевантных документа; отдельные abstention-кейсы проверяют корректный отказ без фиктивных источников. Для критических кейсов дополнительно заданы точный текст доказательства, физическая PDF-страница и номер страницы внутри руководства.

Embedding: `offline-lexical-hash-v2`, размерность `1024`. Прогон выполнен полностью локально с детерминированным лексическим embedding (`--offline-smoke`); документы не передавались внешнему провайдеру.

## Сравнение стратегий

| Стратегия | Коллекция | Chunks | Chunks/док. | Средняя длина, токены | Hit Rate@5 | Evidence Hit@5 | Evidence MRR@10 | Recall@10 | Retrieval, мс |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed-512-64-k10` | `docs_fixed` | 3689 | 368.9 | 321.21 | 0.897 | 0.500 | 0.167 | 0.759 | 38.03 |
| `recursive-512-64-k10` | `docs_recursive` | 3698 | 369.8 | 303.46 | 0.931 | 0.000 | 0.000 | 0.759 | 37.54 |
| `semantic-p95-buffer1-k10` | `docs_semantic` | 3933 | 393.3 | 273.54 | 0.931 | 0.000 | 0.000 | 0.759 | 35.48 |

По прежнему приоритету Hit Rate@5 → MRR@10 → Recall@10 → latency лучшей базовой стратегией стала **`recursive`**. Evidence-метрики показаны как отдельный строгий сигнал и не переключают production chunking по офлайн hash-embedding без сопоставимого прогона основной модели.

## Re-ranker

| Вариант | Hit Rate@5 | Evidence Hit@5 | Evidence MRR@10 | Recall@10 | Retrieval, мс |
|---|---:|---:|---:|---:|---:|
| `without-reranker` | 0.931 | 0.000 | 0.000 | 0.759 | 24.32 |

Re-ranker в этом прогоне отключён флагом `--skip-reranker`; строку до/после необходимо получить локальным прогоном с установленной моделью.

## Подбор параметров

| Стратегия | Коллекция | Chunks | Chunks/док. | Средняя длина, токены | Hit Rate@5 | Evidence Hit@5 | Evidence MRR@10 | Recall@10 | Retrieval, мс |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `recursive-256-32-k10` | `docs_recursive_256_32` | 8360 | 836.0 | 133.15 | 0.828 | 0.000 | 0.000 | 0.753 | 55.3 |
| `recursive-512-64-k10` | `docs_recursive` | 3698 | 369.8 | 303.46 | 0.931 | 0.000 | 0.000 | 0.759 | 23.33 |
| `recursive-512-64-k20` | `docs_recursive` | 3698 | 369.8 | 303.46 | 0.931 | 0.000 | 0.000 | 0.759 | 24.32 |
| `recursive-768-96-k20` | `docs_recursive_768_96` | 2742 | 274.2 | 408.03 | 0.931 | 0.500 | 0.250 | 0.759 | 23.97 |

## Итоговый выбор

Сохраняю **`recursive`** с `chunk_size=512`, `overlap=64`, `top-K=10` как production baseline: Hit Rate@5=0.931, MRR@10=0.756, Recall@10=0.759 и средняя retrieval latency 23.33 мс. Evidence Hit@5=0.000 показывает, что один dense top-10 и document-level labels недостаточны для новых регрессионных вопросов; это не замалчивается сменой конфигурации по локальному hash-embedding, а проверяется отдельным production pipeline eval с dual-query, lexical ranking, ответом и цитатами.

При одинаковых quality-метриках у top-K=10 и top-K=20 выбран top-K=10: он передаёт в следующий этап вдвое меньше кандидатов, а разница локальной latency находится в пределах шума единичного прогона.

Целевой порог Hit Rate@5 ≥ 0.5 выполнен.

Параметры зафиксированы в `app/core/config.py` и `.env.example`. JSON с полными числами находится в `homework_5_4/results/chunking_experiment.json`.

## Ограничения

Офлайн lexical-hash embedding нужен для воспроизводимой проверки pipeline и не заменяет финальный прогон с выбранной embedding-моделью. Числа latency включают локальный in-memory Qdrant и не являются production benchmark.

## Реализация в проекте

В [chunking.py](../app/services/chunking.py) реализованы сравниваемые стратегии,
в [evaluate_chunking.py](../scripts/evaluate_chunking.py) — запуск опыта и
сбор метрик. Таблицы выше относятся к датированному offline-эксперименту.
Текущий структурный splitter и ограничения PDF/CHM описаны в
[архитектурных приёмах](architecture_patterns.md); semantic-разбиение не
объявляется default принятого индекса.

## Воспроизведение

```powershell
python scripts/evaluate_chunking.py --offline-smoke --skip-reranker
python scripts/evaluate_chunking.py --offline-smoke --reranker-local-files-only
```

Второй запуск выполняет локальный BGE re-ranking. Для облачного embedding требуется явное разрешение на передачу фрагментов документации и флаг `--allow-external-data-transfer`.
