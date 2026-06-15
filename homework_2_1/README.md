# Homework 2.1 - API провайдеров

Цель: показать базовую интеграцию Support Copilot с LLM API через OpenAI SDK и
OpenAI-compatible endpoint.

## Что реализовано

- `provider_api_demo.py` подключается к API через OpenAI SDK.
- `.env` загружается через `python-dotenv`.
- API-ключи берутся только из окружения.
- Базовый запрос использует `system` и `user` сообщения.
- Temperature experiment запускает запросы с `0`, `0.7`, `1.5`.
- Streaming-запрос использует `stream=True` и печатает ответ по частям.
- Провайдер переключается через `LLM_PROVIDER=openai` или `LLM_PROVIDER=ollama`.
- Для обычного и streaming-запроса выводятся TTFT и полное время.

## Подготовка

Из корня `HelpDesk_Assistant`:

```powershell
python -m pip install openai python-dotenv
```

Заполнить локальный `.env`:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=<актуальная модель из официальной документации OpenAI>
```

Для локального Ollama-режима:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_API_KEY=ollama
```

Скрипт сам добавляет `/v1` к `OLLAMA_BASE_URL`, чтобы использовать
OpenAI-compatible API Ollama.

## Запуск

OpenAI:

```powershell
python .\homework_2_1\provider_api_demo.py --provider openai
```

Ollama:

```powershell
python .\homework_2_1\provider_api_demo.py --provider ollama
```

Переопределить текст пользовательского запроса:

```powershell
python .\homework_2_1\provider_api_demo.py --prompt "Пользователь не может активировать лицензию после переустановки Windows."
```

Быстро проверить только streaming:

```powershell
python .\homework_2_1\provider_api_demo.py --provider openai --only-stream --stream-debug --stream-debug-limit 12 --max-tokens 120
```

Если streaming работает, ответ начнет печататься до строки итоговой статистики,
а `--stream-debug` после ответа покажет отдельный блок `Stream diagnostics`.
Если `stream events` больше 1, но `text chunks` равно 0, провайдер присылает
служебные события без текстового delta. Если `text chunks` равно 1, код
стриминг использует, но конкретный провайдер или модель отдали текст одним
крупным чанком.

## Наблюдения по temperature

Комментарии для отчета после запуска:

- `temperature=0` должна давать самый стабильный и повторяемый ответ.
- `temperature=0.7` обычно дает более естественную формулировку без сильной
  потери структуры.
- `temperature=1.5` повышает вариативность и может добавить лишние детали, что
  рискованно для техподдержки без проверенной базы знаний.

После фактического запуска нужно заменить эти предварительные наблюдения на
сравнение реальных ответов выбранной модели.

## Проверенные источники

На 2026-06-15 сверены официальные материалы OpenAI:

- Chat Completions API: `POST /chat/completions`, messages и streaming events.
- OpenAI Python SDK: пример `client.chat.completions.create(...)`, `.env` для
  ключа, timeouts и обработка ошибок SDK.

Конкретный OpenAI `model id` не выбран в коде намеренно: по правилам проекта его
нужно задавать через `.env` после проверки актуальной документации и доступности
в аккаунте.
