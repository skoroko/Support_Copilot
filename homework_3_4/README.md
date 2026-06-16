# Домашнее задание 3.4: FastAPI LLM Service

В этом инкременте Support Copilot получил отдельный FastAPI-сервис
`app.main:app`. Сервис оборачивает OpenAI-compatible chat completions,
поддерживает обычный JSON-ответ, SSE streaming, health check, статический список
моделей и cache-aside через Redis-compatible backend.

## Запуск

Установить зависимости:

```powershell
pip install -r requirements.txt
```

Заполнить локальный `.env`:

```text
OPENAI_API_KEY=
APP_DEFAULT_MODEL=gpt-5.4-mini
APP_REDIS_URL=redis://localhost:6379/0
APP_CACHE_TTL_SECONDS=3600
APP_CORS_ORIGINS=*
```

Запустить сервис:

```powershell
uvicorn app.main:app --reload --port 8000
```

Если `OPENAI_API_KEY` не задан, приложение должно падать на старте. Это
ожидаемое поведение для LLM-сервиса. `/health` не проверяет Redis или OpenAI и
после успешного старта работает даже при недоступном Redis.

## Проверочные запросы

Health:

```powershell
curl http://localhost:8000/health
```

Models:

```powershell
curl http://localhost:8000/models
```

Chat:

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Как проверить срок действия лицензии?\"}],\"temperature\":0.2,\"max_tokens\":500}"
```

Streaming:

```powershell
curl -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Сформулируй короткий ответ клиенту.\"}],\"temperature\":0.2,\"max_tokens\":500}"
```

## Что реализовано

- `GET /health` возвращает `{"status":"ok"}` без обращения к Redis/OpenAI.
- `GET /models` возвращает статический список OpenAI-моделей и цен.
- `POST /chat` валидирует сообщения, модель, temperature и max_tokens, вызывает
  `LLMService.complete`, сохраняет ответ в cache-aside и возвращает флаг
  `cached`.
- `POST /chat/stream` возвращает Server-Sent Events: текстовые chunks,
  финальный usage event и `data: [DONE]`.
- `app.main` создает `AsyncOpenAI` и cache в lifespan, закрывает ресурсы при
  shutdown, добавляет CORS, request id, duration logging и exception handlers.
- `app.deps.providers` отдает зависимости через `Annotated`/`Depends` без
  глобальных клиентов.

## Локальная проверка

```powershell
python -m compileall app support_copilot homework_3_4 tests
python -m unittest
```
