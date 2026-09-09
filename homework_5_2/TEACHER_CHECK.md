# Проверка ДЗ 5.2

1. `docker compose up -d qdrant` и `docker compose ps qdrant` — сервис healthy,
   dashboard на `http://127.0.0.1:6333/dashboard`.
2. Дважды выполнить `scripts/load_to_qdrant.py --url http://127.0.0.1:6333
   --offline-smoke`: оба раза `points_count=120`.
3. Проверить `app/services/vector_store.py`: `AsyncQdrantClient`,
   `ensure_collection`, `upsert`, `search`, проверка размерности и payload indexes.
4. Открыть `docs/vector_store.md`: таблица пяти запросов Cosine/Dot и три
   фильтра с фактическим top-3.
5. Запустить `python -m pytest`: dimension mismatch, batching, last-batch wait,
   `query_points`, корпус и Compose проверяются offline.

Основные артефакты: `compose.yaml`, `.env.example`, `app/core/config.py`,
`app/services/vector_store.py`, `scripts/load_to_qdrant.py`,
`scripts/vector_store_experiments.py`, `data/support_documents.json`.
