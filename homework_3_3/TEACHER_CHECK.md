# Краткая сводка для проверки ДЗ 3.3

В этом инкременте в проект `Support Copilot` добавлена асинхронная обработка
LLM-запросов: новый `AsyncLLMClient`, batch-вызовы с ограничением конкурентности,
streaming через async-генератор, FastAPI SSE endpoint и benchmark sync vs async.

Главная проверяемая логика находится в основном пакете `support_copilot`, а не в
одноразовом скрипте. Папка `homework_3_3` содержит краткую инструкцию для сдачи.

## Как проверить

Unit-тесты без реальных API-вызовов:

```powershell
python -m unittest
```

Проверка компиляции:

```powershell
python -m compileall support_copilot scripts tests
```

Benchmark sync vs async в demo mode:

```powershell
python scripts\benchmark.py
```

SSE endpoint:

```powershell
uvicorn support_copilot.api:app --reload --port 8000
curl -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d "{\"prompt\":\"Что такое event loop?\"}"
```

## Что должно быть видно

В benchmark на 20 запросах есть четыре режима:

1. `sync sequential` - старый последовательный baseline.
2. `async batch_chat` с `concurrency=1`.
3. `async batch_chat` с `concurrency=5`.
4. `async batch_chat` с `concurrency=10`.

Проверенный demo-результат:

```text
sync sequential: 2.191s
async batch_chat concurrency=5: 0.440s, speedup 4.98x
async batch_chat concurrency=10: 0.222s, speedup 9.86x
```

В SSE-ответе должны приходить события:

- `event: meta` - время до первого токена;
- `event: delta` - потоковые дельты ответа;
- `event: done` - завершение стрима и общее время.

## Ключевые файлы

- `homework_3_3/README.md` - команды запуска и краткое описание сдачи.
- `support_copilot/infrastructure/async_llm.py` - `AsyncLLMClient`, demo async provider, fallback chain.
- `support_copilot/api.py` - FastAPI app и `POST /chat/stream`.
- `scripts/benchmark.py` - benchmark sync sequential vs async batch.
- `scripts/benchmark_results.md` - лог последнего demo benchmark.
- `support_copilot/config.py` - async-настройки через `.env`.
- `.env.example` - шаблон async-переменных без секретов.
- `pyproject.toml` и `uv.lock` - декларация и воспроизводимый граф FastAPI/SSE/async stack.
- `tests/test_async_llm.py` - тесты async-клиента.
- `tests/test_api.py` - тест FastAPI/SSE без внешней сети.

## Соответствие критериям самопроверки

| Критерий | Где смотреть |
| --- | --- |
| Используется async-клиент SDK | `AsyncOpenAI` в `support_copilot/infrastructure/async_llm.py` |
| Нет синхронного `OpenAI`, `requests`, `time.sleep` внутри async def | `async_llm.py`, проверка `rg "time\.sleep\|requests\|\bOpenAI\(" support_copilot\infrastructure\async_llm.py` |
| `Semaphore` - атрибут экземпляра | `self._sem = asyncio.Semaphore(concurrency)` в `AsyncLLMClient.__init__` |
| `Semaphore` не создается в `complete` / `batch_chat` | тест `test_semaphore_is_instance_attribute_not_method_local` |
| SDK timeout | `AsyncOpenAI(timeout=config.timeout_sec)`; по умолчанию `ASYNC_LLM_SDK_TIMEOUT_SEC=30` |
| Business timeout вокруг `complete` | `_await_with_business_timeout(...)`, `ASYNC_LLM_BUSINESS_TIMEOUT_SEC=15` |
| SDK retry | `AsyncOpenAI(max_retries=self.sdk_max_retries)`, default `ASYNC_LLM_MAX_RETRIES=3` |
| `batch_chat` использует `asyncio.gather(..., return_exceptions=True)` | `AsyncLLMClient.batch_chat()` |
| `concurrency` не является тихим no-op | `batch_chat()` требует совпадения с instance semaphore; benchmark создает отдельный клиент для `1 / 5 / 10` |
| Ошибка одного запроса не роняет весь batch | тест `test_batch_chat_returns_exception_in_failed_position` |
| `stream_chat` возвращает async iterator с дельтами | `AsyncLLMClient.stream_chat()` и тест `test_stream_chat_yields_before_full_answer_finishes` |
| Usage стрима логируется после закрытия | `llm.stream.closed` в `_stream_with_fallback()` |
| SSE endpoint реализован | `support_copilot/api.py`, endpoint `POST /chat/stream` |
| Benchmark показывает ускорение async | `scripts/benchmark_results.md`, speedup `9.86x` при `concurrency=10` |
| API-ключи не в коде | `.env.example` содержит только пустые шаблоны и env-настройки |

## Проверочные команды

```powershell
python -m unittest
python -m compileall support_copilot scripts tests
python scripts\benchmark.py
uvicorn support_copilot.api:app --reload --port 8000
```

Локально проверено:

- `python -m unittest` - 13 tests OK.
- `python -m compileall support_copilot scripts tests` - без ошибок.
- `python scripts\benchmark.py` - async ускорение больше 4-5x при `concurrency=10`.

## Примечание по Python

В задании указан Python 3.11+ для `asyncio.timeout`, `TaskGroup` и
`ExceptionGroup`. На текущей машине доступен Python 3.10.5, поэтому в
`AsyncLLMClient` сделан совместимый wrapper: на Python 3.11+ используется
`asyncio.timeout`, а на Python 3.10 - `asyncio.wait_for`. Основная архитектура
при этом остается той же.
