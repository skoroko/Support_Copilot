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

Полная таблица и вывод доступны в
[историческом отчёте](historical/2026-09-09_chunking_fb594ad.md), машинные
результаты — в [восстановленном JSON](historical/2026-09-09_chunking_fb594ad.json).

## Исторические результаты

[Отчёт от 9 сентября 2026 года](historical/2026-09-09_chunking_fb594ad.md) и
[исходный JSON](historical/2026-09-09_chunking_fb594ad.json) восстановлены
без изменений из `fb594ad` (`docs/chunking_experiment.md` и
`homework_5_4/results/chunking_experiment.json`). Пути внутри снимка относятся
к корню проекта того коммита. Это исторический lexical-hash эксперимент,
включавший тогда локальный BGE; он не подтверждает наличие BGE сейчас.
