# Домашнее задание 5.2 — Qdrant

Реализовано подключение Qdrant к накопительному проекту Support Copilot:

- Docker-сервис Qdrant 1.19 с API key, named volume и TCP healthcheck;
- Pydantic-настройки URL, ключа, коллекции и размерности;
- один async-клиент в FastAPI lifespan;
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
