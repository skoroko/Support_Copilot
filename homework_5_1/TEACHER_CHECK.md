# Проверка ДЗ 5.1

1. `tests/eval/mini_benchmark.json` - 8 предметных пар, формат каждого элемента
   строго `query`, `relevant`, `irrelevant`.
2. `app/services/embeddings.py` экспортирует
   `embed_texts(list[str]) -> list[list[float]]`.
3. `EMBEDDING_BATCH_SIZE=100`, допустимый минимум валидации - 32.
4. Все векторы L2-нормализуются; размерность ответа проверяется.
5. Retry: только 429, timeout, connection errors и 5xx; unit-тест подтверждает,
   что HTTP 400 не повторяется.
6. Кеш SQLite переживает новый экземпляр сервиса. Unit-тест подтверждает один
   provider call до рестарта и ноль после него.
7. Модель и dimensions входят в cache key. Unit-тест подтверждает cache miss при
   `model-small -> model-large`.
8. Второй локальный CLI-прогон: 24 cache hits, 0 provider-like requests; результат
   сохранён в `homework_5_1/results/cache_smoke.json`.
9. Выбран `text-embedding-3-large`, `dimensions=1024`: приоритет русскоязычного
   retrieval, малая абсолютная стоимость корпуса и меньшее vector storage.
10. Для 60 документов по 20 страниц расчёт составляет $0.1248 синхронно или
    $0.0624 через Batch API. Скрипт: `python homework_5_1/cost_estimate.py`.

Команда проверки:

```powershell
python -m pytest tests/unit/test_embeddings.py -q -p no:cacheprovider
python homework_5_1/cost_estimate.py
python homework_5_1/benchmark.py --offline-smoke
```

Реальный OpenAI benchmark намеренно отделён от offline smoke, потому что он
передаёт локальные учебные фрагменты внешнему провайдеру. Команда и условия
запуска описаны в `homework_5_1/README.md`.
