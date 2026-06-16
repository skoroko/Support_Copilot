# Краткая сводка для проверки ДЗ 3.4

В этом инкременте реализован отдельный FastAPI LLM Service для проекта
`Support Copilot`. Основная точка входа — `app.main:app`, запуск:

```powershell
uvicorn app.main:app --reload --port 8000
```

Сервис построен как production-oriented слой поверх OpenAI-compatible chat API:
на старте через lifespan создаются `AsyncOpenAI` и Redis-compatible cache, в
эндпоинтах используется FastAPI dependency injection, а ошибки провайдера
переводятся в стабильный HTTP-контракт.

## Как проверить

Без обращения к LLM:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/models
```

Обычный chat-запрос:

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Как проверить срок действия лицензии?\"}],\"temperature\":0.2,\"max_tokens\":500}"
```

Streaming:

```powershell
curl -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Сформулируй короткий ответ клиенту.\"}],\"temperature\":0.2,\"max_tokens\":500}"
```

## Что должно быть видно

`/health` возвращает `{"status":"ok"}` и не зависит от Redis/OpenAI. `/models`
возвращает статический список моделей с ценами. `/chat` возвращает JSON
`content`, `model`, `usage`, `finish_reason`, `cached`. При повторе одинакового
запроса должен вернуться `cached: true`, потому что `LLMService` использует
cache-aside ключ `chat:` + SHA-256 от содержимого запроса без `user_id`,
`session_id` и `stream`.

`/chat/stream` отдает SSE chunks в формате `data: <text>\n\n`, затем usage event
`data: {"usage": ...}\n\n` и завершает поток `data: [DONE]\n\n`.

## Ключевые файлы

- `app/main.py` - FastAPI app, lifespan, CORS, middleware, exception handlers.
- `app/core/config.py` - настройки через `pydantic-settings`, `.env`, `SecretStr`.
- `app/core/exceptions.py` - доменные LLM-исключения.
- `app/deps/providers.py` - dependency injection providers.
- `app/routers/chat.py` - `/chat` и `/chat/stream`.
- `app/routers/health.py` - `/health`.
- `app/routers/models.py` - `/models`.
- `app/services/llm.py` - `LLMService`, cache-aside, streaming, mapping ошибок.
- `app/schemas/chat.py` - Pydantic-схемы запросов/ответов.
- `app/schemas/models.py` - статический список моделей и цен.
- `.env.example` - переменные без секретов.
- `tests/test_app_fastapi.py` - offline-проверка FastAPI endpoints.
- `tests/test_app_llm_service.py` - offline-проверка cache и streaming logic.

## Соответствие критериям самопроверки

| Критерий | Где смотреть |
| --- | --- |
| `uvicorn app.main:app` запускает сервис | `app/main.py` |
| Lifespan создает `AsyncOpenAI` и Redis cache | `app/main.py` |
| Настройки через `pydantic-settings` v2 и `.env` | `app/core/config.py`, `.env.example` |
| 4 endpoint: `/chat`, `/chat/stream`, `/health`, `/models` | `app/routers/*.py` |
| ChatRequest/ChatResponse с Pydantic-валидацией | `app/schemas/chat.py` |
| SSE streaming с usage и `[DONE]` | `app/routers/chat.py`, `app/services/llm.py` |
| `/health` без внешних зависимостей | `app/routers/health.py` |
| Статический список моделей/цен | `app/schemas/models.py` |
| DI без глобальных клиентов | `app/deps/providers.py` |
| Cache-aside и SHA-256 key | `app/services/llm.py` |
| Mapping OpenAI ошибок в доменные ошибки | `app/services/llm.py`, `app/core/exceptions.py` |
| Request id, duration logging, CORS | `app/main.py` |
| OpenAPI examples/responses | `app/schemas/chat.py`, `app/routers/*.py` |
| Offline-тесты без сетевых вызовов | `tests/test_app_fastapi.py`, `tests/test_app_llm_service.py` |

## Проверочные команды

```powershell
python -m compileall app support_copilot homework_3_4 tests
python -m unittest
uvicorn app.main:app --reload --port 8000
```
