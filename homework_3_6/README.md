# Домашнее задание 3.6: Observability ИИ-приложений

В этом инкременте к Docker-стеку FastAPI LLM Service добавлен observability-слой:
Phoenix UI, OpenAI auto-instrumentation, JSON-логи с `request_id` и безопасное
PII-маскирование prompt preview.

## Запуск

```powershell
docker compose up -d --build
```

Phoenix UI:

```text
http://localhost:6006
```

FastAPI:

```text
http://localhost:8000/docs
```

## Проверка логов

```powershell
docker compose logs app
```

В строках логов должны быть:

- `event`;
- `request_id`;
- `path`;
- `method`;
- `latency_ms`;
- для LLM-вызова: `model`, `input_tokens`, `output_tokens`, `finish_reason`,
  `prompt_hash`, `prompt_preview`.

`prompt_preview` должен содержать плейсхолдеры `[EMAIL]`, `[PHONE_RU]`,
`[CARD]`, `[INN]`, `[PASSPORT]` вместо исходных PII.

## Локальные проверки

```powershell
python -m compileall app support_copilot homework_3_6 tests
python -m unittest
docker compose config --quiet
```
