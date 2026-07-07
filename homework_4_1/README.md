# ДЗ 4.1 - архитектура чата и хранение истории

## Что реализовано

- `app/chat/domain.py` - доменные Pydantic v2 модели `Chat` и `ChatMessage`.
- `app/chat/repository.py` - `ChatRepository` как `typing.Protocol`.
- `app/chat/repositories/json_repo.py` - append-only JSONL-хранилище с soft-delete marker.
- `app/chat/repositories/pg_models.py` и `pg_repo.py` - async SQLAlchemy 2.x модели и Postgres-репозиторий.
- `app/chat/service.py` - `ChatService`, sliding window context strategy, `count_tokens()` и `fit_to_budget()`.
- `app/chat/routes.py` - endpoints `POST /chats`, `GET /chats/{id}`, `POST/GET/DELETE /chats/{id}/messages`.
- `app/chat/deps.py` - DI-фабрика репозитория через `CHAT_REPOSITORY=json|postgres`.
- Stateful `POST /chats/{id}/messages` использует входной security validator из M3.
- `alembic/versions/20260707_0001_chat_tables.py` - миграция таблиц `chats` и `chat_messages`.
- `docs/chat.md` - Mermaid-диаграмма, стратегия контекста, curl-примеры и переключение хранилищ.

## Проверка

```powershell
python -m pytest tests/chat
python -m pytest
python -m compileall app alembic tests homework_4_1
```

Postgres contract-тесты используют тот же набор проверок, что и JSON, но запускаются
только если задан `CHAT_TEST_DATABASE_URL`. Без этой переменной они пропускаются,
чтобы обычная локальная проверка не требовала поднятой БД.

## Запуск JSON-режима

```powershell
$env:CHAT_REPOSITORY='json'
$env:CHAT_STORAGE_DIR='./var/chats'
uvicorn app.main:app --reload
```

## Запуск Postgres-режима

```powershell
docker compose up -d postgres
$env:CHAT_REPOSITORY='postgres'
$env:CHAT_DATABASE_URL='postgresql+asyncpg://support:support@localhost:5432/support_copilot'
alembic upgrade head
uvicorn app.main:app --reload
```
