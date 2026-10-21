# Домашнее задание 5.2 — Qdrant

## Проверка на реальном корпусе — 21 октября 2026

[Отчёт](REAL_INDEX_2026-10-21.md) и
[метрики](results/real_index_2026-10-21.json) фиксируют опыт с **100 реальными
фрагментами** из действующего индекса: две изолированные коллекции Cosine/Dot,
повторный upsert, HNSW и фильтры по метаданным. Исходные векторы использованы
повторно; embeddings вызваны только для пяти поисковых вопросов. Рабочий индекс
содержал 14 139 точек до и после опыта. Тексты фрагментов и векторы в отчёт не
включены.

## Исходный учебный инкремент

Описание ниже относится к первоначальному опыту со 120 синтетическими
документами `data/support_documents.json` и offline lexical-hash embeddings.
Этот материал не включается в действующий RAG-корпус и не подтверждает качество
поиска на реальных мануалах. Текущий опыт выше использует фактические поля
`product`, `source_id`, `page`; первоначальные category/date/tenant-фильтры
сохранены как история учебной реализации.

Реализовано подключение Qdrant к накопительному проекту Support Copilot:

- Docker-сервис Qdrant 1.19 с API key, named volume и TCP healthcheck;
- настройки URL, ключа, коллекции и размерности;
- один async-клиент, переиспользуемый внутри каждого скрипта (на момент сдачи —
  в FastAPI lifespan; после аудита F20 приложение оставило единственный store
  внутри `RAGService`, а `VectorStore` остался адаптером этих скриптов);
- идемпотентная загрузка 120 предметных документов с UUID v5;
- Cosine-коллекция 1024 dimensions, HNSW `m=16`, `ef_construct=100`;
- шесть payload indexes, vector search и три вида фильтров;
- воспроизводимый Cosine/Dot эксперимент с удалением временных коллекций.

## Запуск

```powershell
Copy-Item .env.example .env
# Заменить все демонстрационные секреты, включая QDRANT_API_KEY.
docker compose up -d qdrant
docker compose ps qdrant
```

Для команды с хоста URL должен указывать на опубликованный порт:

```powershell
python scripts/load_to_qdrant.py --url http://127.0.0.1:6333 --offline-smoke
python scripts/load_to_qdrant.py --url http://127.0.0.1:6333 --offline-smoke
python scripts/vector_store_experiments.py --url http://127.0.0.1:6333 --offline-smoke
```

Если API key не подхватывается из `.env`, передайте его через `--api-key`.
Для реального OpenAI embeddings уберите `--offline-smoke`; перед этим заполните
`OPENAI_API_KEY`, проверьте `EMBEDDING_DIMENSIONS=1024` и разрешение на передачу
текстов провайдеру.

Полное описание архитектуры, таблица Cosine/Dot и результаты фильтров находятся
в `docs/vector_store.md`. Машиночитаемый прогон — в
`homework_5_2/results/experiment_offline.json`.

## Проверки

```powershell
python -m pytest tests/unit/test_vector_store.py tests/unit/test_qdrant_loader.py
python -m pytest
docker compose config --quiet
```
