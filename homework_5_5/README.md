# Домашнее задание 5.5 — корпоративный RAG-ассистент

В этом инкременте собраны ingestion, guarded query, multi-turn, citations,
загрузка документов, Telegram streaming и feedback с источниками.

## Быстрая проверка без внешних вызовов

```powershell
python -m pytest -q -p no:cacheprovider
python scripts/ingest.py data/ --offline-smoke --collection homework_5_5_offline --dimensions 64 --docstore-path var/homework_5_5_docstore.json --offline-qdrant-path var/homework_5_5_qdrant
python scripts/ingest.py data/ --offline-smoke --collection homework_5_5_offline --dimensions 64 --docstore-path var/homework_5_5_docstore.json --offline-qdrant-path var/homework_5_5_qdrant
docker compose config --quiet
```

Ожидаемый итог второго ingestion: `0 changed, 1992 unchanged, 0 failed`.

## Основные результаты

- 70 реальных файлов: 9 PDF, 60 HTML и 1 CHM;
- 1992 document units и 3508 чанков при `recursive/512/64`;
- `IngestionPipeline` + persisted `SimpleDocumentStore` + `UPSERTS`;
- retrieval `top_k=10`, optional BGE `top_n=5`, score guard до LLM;
- источники `id/file_name/page/score/snippet/confident` и цитаты `[1]`;
- multi-turn rewrite с Postgres history;
- SSE `event: sources`, Telegram edit debounce 0.8 секунды;
- upload 202 + background indexing;
- feedback хранит показанные источники.

Перед внешним ingestion требуется явный opt-in
`RAG_ALLOW_EXTERNAL_DATA_TRANSFER=true`. Первый Docker-прогон 10 сентября
2026 года получил от OpenAI `403 unsupported_country_region_territory`, при
этом API корректно запустился в degraded-режиме. Повтор с Windows-хоста через
доступный сетевой маршрут завершился успешно: `1992 changed`, 3508 чанков,
0 ошибок; повторный запуск дал `0 changed, 1992 unchanged`. Контрольный запрос
успешно прошёл embeddings, Qdrant retrieval и chat completion. Полный
офлайн-прогон выше позволяет проверить ingestion без отправки данных в сеть.
