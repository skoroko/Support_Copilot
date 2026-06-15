# Домашнее задание 3.3: асинхронная обработка запросов к ИИ

В этом инкременте синхронный подход из предыдущих модулей дополнен
асинхронным LLM-клиентом и SSE endpoint для потоковой выдачи ответа.

## Что реализовано

- `AsyncLLMClient` поверх `AsyncOpenAI`.
- Методы `complete`, `batch_chat`, `stream_chat`.
- `self._sem` как instance-level `asyncio.Semaphore`, создается один раз в `__init__`.
- SDK retry через `AsyncOpenAI(max_retries=3)`.
- SDK timeout через `AsyncOpenAI(timeout=30)`.
- Business timeout: `asyncio.timeout(15)` на Python 3.11+, fallback на `asyncio.wait_for` для текущего локального Python 3.10.
- `batch_chat(..., return_exceptions=True)` не роняет весь батч из-за одного запроса.
- `stream_chat` отдает дельты по мере прихода и логирует финальный usage.
- FastAPI SSE endpoint `POST /chat/stream`.
- Benchmark sync sequential vs async concurrency `1 / 5 / 10`.

## Основные файлы

- `support_copilot/infrastructure/async_llm.py` - async-клиент и demo async provider.
- `support_copilot/api.py` - FastAPI app и SSE endpoint.
- `scripts/benchmark.py` - benchmark.
- `scripts/benchmark_results.md` - результат локального demo benchmark.
- `tests/test_async_llm.py` - unit-тесты async-клиента без сетевых вызовов.

## Запуск benchmark

```powershell
python scripts/benchmark.py
```

Проверенный demo-результат:

```text
sync sequential: 2.191s
async batch_chat concurrency=5: 0.440s, speedup 4.98x
async batch_chat concurrency=10: 0.222s, speedup 9.86x
```

## Запуск SSE endpoint

```powershell
uvicorn support_copilot.api:app --reload --port 8000
```

Проверка:

```powershell
curl -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d "{\"prompt\":\"Что такое event loop?\"}"
```

Для real mode нужны переменные из `.env`: `OPENAI_API_KEY`/`OPENAI_MODEL`,
`OPENROUTER_API_KEY`/`OPENROUTER_MODEL` или локальная Ollama через `OLLAMA_MODEL`.

## Проверки

```powershell
python -m unittest
python -m compileall support_copilot scripts tests
```

Локально проверено:

- `python -m unittest` - 13 tests OK.
- `python -m compileall support_copilot scripts tests` - без ошибок.
- `python scripts/benchmark.py` - async ускорение больше 4-5x при concurrency=10.
