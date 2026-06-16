# Краткая сводка для проверки ДЗ 3.5

В этом инкременте FastAPI LLM Service из ДЗ 3.4 контейнеризирован. Одна команда
поднимает приложение и Redis:

```powershell
docker compose up -d --build
```

Перед запуском нужно создать локальный `.env` из `.env.example` и заполнить
`OPENAI_API_KEY`. Файл `.env` не коммитится и не копируется в Docker-образ.

## Что проверять

`Dockerfile` сделан multi-stage: builder ставит зависимости через `uv`, runtime
копирует `/app` с venv, создает `appuser` с uid 1000 и запускает `uvicorn` в
exec-форме на `0.0.0.0:8000`.

`compose.yaml` поднимает два сервиса: `app` и `redis`. У `app` есть порт
`8000:8000`, healthcheck на `/ready`, `depends_on` с `condition:
service_healthy`. Redis использует `redis:7.4-alpine`, named volume
`redis_data:/data`, healthcheck `redis-cli ping` и не пробрасывает порт наружу.

В API разделены liveness и readiness:

- `/health` всегда возвращает `200 {"status":"ok"}` и не зависит от Redis.
- `/ready` проверяет Redis через `ping` с таймаутом и возвращает `200` при
  доступном Redis или `503 {"status":"degraded","redis":"down"}` при сбое.

## Команды проверки

```powershell
docker build -t llm-service:v1 .
docker images llm-service
docker run --rm llm-service:v1 ls -la /app

docker compose up -d --build
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/docs
docker compose exec app id
docker compose exec redis redis-cli ping
docker compose stop redis
curl http://localhost:8000/ready
curl http://localhost:8000/health
docker compose down
```

## Ключевые файлы

- `Dockerfile` - multi-stage build, `uv`, slim base, non-root runtime.
- `.dockerignore` - исключение секретов и лишних файлов из build context.
- `compose.yaml` - `app + redis`, healthcheck-и, `depends_on: service_healthy`.
- `.env.example` - имена переменных и плейсхолдеры без реальных секретов.
- `app/routers/health.py` - `/health` и `/ready`.
- `app/main.py` - создание Redis-compatible cache в lifespan.
- `tests/test_docker_config.py` - offline-проверка Docker/Compose контракта.
- `tests/test_app_fastapi.py` - проверка `/health`, `/ready`, `/chat`,
  `/chat/stream`.

## Соответствие критериям самопроверки

| Критерий | Где смотреть |
| --- | --- |
| Multi-stage Dockerfile на slim Python | `Dockerfile` |
| Зависимости ставятся через `uv` до копирования кода | `Dockerfile` |
| Runtime запускается не под root | `Dockerfile`, `USER appuser` |
| CMD exec-form и `--host 0.0.0.0` | `Dockerfile` |
| Build context чистится от секретов и tests | `.dockerignore` |
| Compose поднимает `app` и `redis` | `compose.yaml` |
| `depends_on: service_healthy` | `compose.yaml` |
| Redis без внешнего `ports` | `compose.yaml` |
| Redis named volume | `compose.yaml`, `redis_data` |
| `/health` liveness без внешних зависимостей | `app/routers/health.py` |
| `/ready` readiness с Redis ping | `app/routers/health.py` |
| `.env` не должен быть в git | `.gitignore`, проверка `git ls-files` |

## Проверки без Docker

```powershell
python -m compileall app support_copilot homework_3_5 tests
python -m unittest
docker compose config --quiet
git ls-files | Select-String -Pattern '\.env$'
```
