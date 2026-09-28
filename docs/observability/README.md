# Observability trace notes

Скриншот Phoenix UI для ДЗ 3.6 нужно положить в эту папку после запуска:

```powershell
docker compose up -d --build
curl http://localhost:6006
```

Что должно быть видно на трейсе `/chat`:

- проект `diploma-fastapi`;
- span OpenAI chat completion;
- `gen_ai.request.model`;
- usage-поля `gen_ai.usage.input_tokens` и `gen_ai.usage.output_tokens`;
- latency запроса;
- связь с HTTP-логом через `request_id` в логах `app`.

Логи приложения не содержат сырой prompt: пишутся только `prompt_hash` и
`prompt_preview` после `redact_pii()`.
При `TRACING_ENABLED=true` OpenAI и LlamaIndex instrumentors получают один
явный `TraceConfig`: содержимое inputs/outputs, сообщений, prompts/choices,
изображений, embedding text/vectors и параметров вызова модели скрывается.
Если версия optional tracing extra не поддерживает эту конфигурацию, tracing
не запускается. Токены, latency и другие числовые метрики остаются доступны.

Примечание: в Docker Compose переменная `PHOENIX_COLLECTOR_ENDPOINT` задана как
`http://phoenix:6006`, а приложение нормализует ее до OTLP HTTP endpoint
`http://phoenix:6006/v1/traces`.

Privacy boundary for optional tracing: all `retrieval.documents.*` attributes
(including corpus text, metadata, document IDs, and scores) are omitted. Span
events and exception details are omitted; status codes remain available without
descriptions. This applies to the OpenAI and LlamaIndex instrumentors.

## Артефакт

Скриншот Phoenix UI сохранен в [`phoenix_trace.png`](phoenix_trace.png).
На нем видны проект `diploma-fastapi`, span `ChatCompletion`, статус trace, latency
и token/cost summary. Подробные поля `model`, `input_tokens`, `output_tokens`,
`finish_reason`, `prompt_hash` и безопасный `prompt_preview` дополнительно
проверяются в JSON-логах приложения через `docker compose logs app`.
