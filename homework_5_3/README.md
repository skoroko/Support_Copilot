# Домашнее задание 5.3 — RAG с LlamaIndex и без фреймворка

Реализованы две эквивалентные RAG-цепочки для двуязычной документации GTS NX, отдельный CHM-loader, две коллекции Qdrant и FastAPI endpoint `/rag/query`.

## Быстрая проверка без внешних вызовов

```powershell
python -m pytest tests/unit/test_rag.py -q -p no:cacheprovider
python -m pytest -q -p no:cacheprovider
docker compose config --quiet
```

## Реальный запуск

1. Заполнить `.env`, особенно `OPENAI_API_KEY` и `QDRANT_API_KEY`.
2. Получить разрешение на передачу фрагментов руководств внешнему провайдеру.
3. Запустить Qdrant: `docker compose up -d qdrant`.
4. Выполнить сравнение: `python scripts/evaluate_rag.py --backend all --allow-external-data-transfer`.
5. Для API установить `RAG_ENABLED=true` и запустить приложение.

Пример запроса:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:18000/rag/query `
  -ContentType 'application/json' `
  -Body '{"question":"Какие типы сетки строит AUTO-SOLID?"}'
```

Архитектура, версии, сравнение и протокол проверки описаны в `docs/rag.md`. Согласованный набор вопросов находится в `homework_5_3/questions.json`.
