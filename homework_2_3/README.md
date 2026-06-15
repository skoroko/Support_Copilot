# Homework 2.3 - Reliability

Цель: создать надежного клиента для LLM API с retry, fallback, логированием,
usage/cost tracking и базовой защитой от повторных отказов провайдера.

## Что реализовано

- `robust_llm_client.py` содержит класс `RobustLLMClient`.
- Клиент работает поверх OpenAI SDK и OpenAI-compatible endpoint.
- Retry выполняется только для `429`, `5xx` и timeout.
- Backoff: `1s -> 2s -> 4s -> 8s -> 16s` плюс jitter, до 5 retry-попыток.
- Fallback-цепочка real mode: `OpenAI -> OpenRouter -> Ollama -> "Сервис временно недоступен"`.
- Лог содержит timestamp, provider, code, attempt и задержку перед retry.
- Usage tracking считает `prompt_tokens`, `completion_tokens`, `total_tokens` и
  стоимость по ценам из переменных окружения.
- Бонус: circuit breaker на 3 ошибки подряд с cooldown 60 секунд и простой
  client-side rate limiter.

## Запуск offline demo

Из родительской папки проекта:

```powershell
python .\HelpDesk_Assistant\homework_2_3\demo_retry_fallback.py
```

Из папки `HelpDesk_Assistant`:

```powershell
python .\homework_2_3\demo_retry_fallback.py
```

Offline demo не делает сетевых вызовов: OpenAI и OpenRouter заменены фейковыми
клиентами, которые последовательно возвращают `429`, `500`, timeout и успешный
ответ fallback-провайдера.

## Запуск с реальными провайдерами

Сначала заполнить `.env`. Основная цепочка real mode:

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_MODEL=<актуальная модель из документации провайдера>

OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=<актуальная модель OpenRouter>

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_API_KEY=ollama
```

Затем:

```powershell
python .\homework_2_3\demo_retry_fallback.py --real
```

## Результаты

Лог проверенного offline-запуска находится в `results/retry_fallback_log.txt`.
