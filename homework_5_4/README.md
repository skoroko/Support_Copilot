# Домашнее задание 5.4 - chunking и оптимизация retrieval

Для корпуса GTS NX подготовлен golden dataset на 28 вопросов, реализованы три
стратегии chunking, document-level метрики Hit Rate@5, MRR@10 и Recall@10,
локальный BGE re-ranker и подбор `chunk_size`, `overlap`, `top-K`.

## Быстрая проверка

```powershell
python -m pytest tests/unit/test_chunking_retrieval.py -q -p no:cacheprovider
python scripts/evaluate_chunking.py --offline-smoke --skip-reranker
```

Полный локальный прогон с заранее загруженным BGE:

```powershell
python scripts/evaluate_chunking.py --offline-smoke --reranker-local-files-only
```

Без `--offline-smoke` скрипт требует явный флаг
`--allow-external-data-transfer`, потому что embedding-фрагменты документации
будут отправлены настроенному провайдеру.

## Итог

- базовая стратегия: `recursive`;
- параметры: `chunk_size=512`, `overlap=64`, `top-K=10`;
- Hit Rate@5: `0.929`, MRR@10: `0.771`, Recall@10: `0.750`;
- BGE поднял MRR@10 до `0.839`, но на CPU увеличил среднюю latency до
  `18014.63` мс, поэтому оставлен опциональным этапом.

Полная таблица и вывод находятся в `docs/chunking_experiment.md`, машинные
результаты - в `results/chunking_experiment.json`.
