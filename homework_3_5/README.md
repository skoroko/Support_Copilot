# Домашнее задание 3.5: Docker и контейнеризация

В этом инкременте FastAPI-сервис `app.main:app` из ДЗ 3.4 упакован в Docker и
запускается вместе с Redis через Docker Compose.

## Что добавлено

- `Dockerfile` — multi-stage build на `python:3.13-slim-bookworm`.
- `.dockerignore` — исключает `.env`, `.git`, tests, cache, homework-материалы и
  лишние локальные файлы из build context.
- `compose.yaml` — сервисы `app` и `redis`, healthcheck-и, `depends_on:
  service_healthy`, Redis named volume.
- `/ready` — readiness endpoint с проверкой Redis через `ping`.
- `.env.example` — переменные для Docker-стека без реальных секретов.

## Запуск

```powershell
Copy-Item .env.example .env
# заполнить OPENAI_API_KEY в .env
docker compose up -d --build
```

## Проверка

```powershell
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/docs
docker compose exec app id
docker compose exec redis redis-cli ping
docker images llm-service
docker run --rm llm-service:v1 ls -la /app
```

Ожидаемое поведение:

- `/health` всегда возвращает `200 {"status":"ok"}`.
- `/ready` возвращает `200 {"status":"ok","redis":"up"}`, когда Redis доступен.
- Если остановить Redis командой `docker compose stop redis`, `/ready` должен
  вернуть `503 {"status":"degraded","redis":"down"}`, а `/health` должен остаться
  `200`.
- `docker compose exec app id` должен показать `uid=1000(appuser)`.
- В `/app` внутри образа не должно быть `.env`, `.git`, `tests/`, `__pycache__/`.

## Локальные проверки без Docker

```powershell
python -m compileall app support_copilot homework_3_5 tests
python -m unittest
docker compose config --quiet
git ls-files | Select-String -Pattern '\.env$'
```
