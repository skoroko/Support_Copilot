# Краткая сводка для проверки ДЗ 3.6

В этом инкременте добавлен observability-слой для FastAPI LLM Service.

## Как запустить

```powershell
docker compose up -d --build
```

Проверочные URL:

- FastAPI Swagger: `http://localhost:8000/docs`
- Phoenix UI: `http://localhost:6006`
- Liveness: `http://localhost:8000/health`
- Readiness: `http://localhost:8000/ready`

## Что проверять

Phoenix добавлен в `compose.yaml` как сервис `phoenix`:

- image `arizephoenix/phoenix:latest`;
- ports `6006:6006` и `4317:4317`;
- volume `phoenix-data:/data`;
- env `PHOENIX_WORKING_DIR=/data`.

В `app` задан `PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006`, а трассировка
включается в lifespan до создания `AsyncOpenAI`. В коде endpoint нормализуется до
`/v1/traces`, потому что текущий Phoenix принимает OTLP HTTP traces по этому пути.

Логи приложения структурированные JSON. Middleware создает или принимает
`x-request-id`, биндует его в contextvars и ставит тот же id в response header.
LLM-сервис после ответа пишет событие `llm_request_completed` с полями:

- `model`;
- `input_tokens`;
- `output_tokens`;
- `latency_ms`;
- `finish_reason`;
- `prompt_hash`;
- `prompt_preview`;
- `cached`.

Сырой prompt не логируется. `prompt_preview` проходит через `redact_pii()`.

## Ключевые файлы

- `compose.yaml` - Phoenix service и endpoint для app.
- `app/observability/tracing.py` - `phoenix.otel.register` и OpenAIInstrumentor.
- `app/observability/logging.py` - structlog JSONRenderer и contextvars.
- `app/observability/pii.py` - regex-маскирование EMAIL, PHONE_RU, CARD, INN, PASSPORT.
- `app/main.py` - `setup_logging()`, `setup_tracing()`, request middleware.
- `app/services/llm.py` - событие `llm_request_completed`.
- `docs/observability/README.md` - место для скриншота Phoenix trace и пояснение.
- `tests/test_observability.py` - PII и safe logging tests.

## Проверочные команды

```powershell
python -m compileall app support_copilot homework_3_6 tests
python -m unittest
docker compose config --quiet
docker compose up -d --build
docker compose ps
curl http://localhost:6006
curl http://localhost:8000/health
curl http://localhost:8000/ready
docker compose logs app
```

## Соответствие критериям самопроверки

| Критерий | Где смотреть |
| --- | --- |
| Phoenix поднимается через compose | `compose.yaml` |
| UI Phoenix на `localhost:6006` | `compose.yaml`, ports |
| OpenAI auto-instrumentation | `app/observability/tracing.py` |
| Tracing вызывается до AsyncOpenAI | `app/main.py`, lifespan |
| JSON-логи с `request_id` | `app/observability/logging.py`, `app/main.py` |
| LLM-лог с model/tokens/latency/finish_reason | `app/services/llm.py` |
| Сырой prompt не логируется | `app/services/llm.py` |
| PII заменяется плейсхолдерами | `app/observability/pii.py` |
| Unit-тесты PII | `tests/test_observability.py` |
