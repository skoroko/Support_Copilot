# Архитектурный паспорт Support Copilot

Статус: реализованный Telegram/agent/RAG-контур и исторические ADR.
Последнее обновление: 2026-10-13.

## Область применения

Support Copilot - внутренний ИИ-ассистент для инженеров поддержки MIDAS.
Ассистент помогает искать информацию в базе знаний, готовить черновики ответов,
классифицировать обращения, определять случаи для эскалации и собирать обратную
связь. Он не принимает решения вместо инженера поддержки: финальный ответ
проверяет и отправляет человек.

Текущие интерфейсы — CLI и Telegram через FastAPI. Telegram всегда использует
persistent agent внутри существующего chat facade; переключателя режима нет.
CLI и прямые RAG API сохраняют собственные пути.

## Текущий Telegram-контур

```mermaid
flowchart LR
    User[Инженер в Telegram] --> Bot[aiogram: admission и per-user chat binding]
    Bot --> Chat[FastAPI /chats: auth, budgets, input/media moderation]
    Chat --> Service[ChatService: turn lock, output moderation, persistence]
    Service --> Adapter[ChatAgent: private owner + chat thread]
    Adapter --> Graph[Persistent graph: bounded context, ReAct tools]
    Graph --> Search[search_knowledge_base: evidence only]
    Search --> RAG[RAGService.retrieve: production retrieval gates]
    RAG --> Manuals[(Qdrant: production manual corpus)]
    Graph --> Saver[(SQLite or existing Postgres checkpoints)]
    Graph --> Preview[Durable exact recipient + text preview]
    Preview --> Buttons[Telegram Confirm / Cancel]
    Buttons --> Approval[Owned chat + exact draft ID + shared turn lock]
    Approval --> Graph
    Graph -->|approved only| Notifier[Internal notifier + bot idempotency guard]
    Service --> History[(JSON or Postgres history + cited sources + feedback)]
    Service --> Bot
```

`ChatAgent` — адаптер графа, а не второй chat transport. Все Telegram-запросы
сохраняют существующие multipart extraction, admission, moderation,
request-budget lease и feedback. RAG service инъецируется в graph лениво из
`app.state`; отдельный индекс или synthetic fallback не создаётся. Public
`POST /rag/retrieve` возвращает те же выбранные evidence fragments без answer
generation. Несколько поисков используют стабильные citation numbers, а
финальные sources проходят проверку по реально возвращённым фрагментам.

В checkpoint хранится максимум пять human-turn groups и 24 tool results;
provider context дополнительно подчиняется `CHAT_CONTEXT_WINDOW` и token
budget `CHAT_MODEL_CONTEXT_TOKENS - CHAT_RESPONSE_TOKENS -
CHAT_SAFETY_MARGIN_TOKENS`. При переполнении текущей группы graph возвращает
явный отказ без модельного вызова. Raw image payload не сохраняется; captions,
transcripts и document text доступны агенту. У старого chat history graph
восстанавливает ограниченный контекст при пустом checkpoint.

Telegram owner/interface разрешается сервером в один durable chat, включая
перезапуск бота. Approval принадлежит точному owner + chat + draft UUID.
Callback не может изменить recipient/text, повторить исполненное действие или
одобрить более новый draft. `/clear` удаляет и chat history, и checkpoint под
тем же turn lock. Send/cancel/unknown delivery status формируется из observed
tool result без следующего LLM-вызова. Таймаут/disconnect оставляет request
budget занятым, пока retrieval worker реально не завершится.

Chat repository и checkpoint saver имеют отдельные транзакции. При аварии
между graph completion и chat append history может отставать; это ограничение
не следует трактовать как общую exactly-once транзакцию. Delivery защищена
существующим notifier/bot idempotency guard; неизвестный исход отправки требует
проверки у получателя. Подробности и offline check: [unified_agent.md](unified_agent.md).

## Исторический эскиз CLI (домашнее задание 3.2)

Следующие исходные слои и ADR фиксируют учебный дизайн июня 2026 года. Метки
«будущее», JSON knowledge base и API «позже» относятся к этому историческому
эскизу; текущий Telegram-контур описан выше.

## Архитектура из четырех слоев

```mermaid
flowchart LR
    User[Инженер поддержки] -->|1. вопрос| Gateway

    subgraph G["Gateway layer"]
        Gateway[CLI сейчас / FastAPI позже]
        Auth[будущее: auth, rate limit, request id]
    end

    subgraph S["Service layer"]
        Orchestrator[Оркестратор диалога]
        Prompt[Загрузка prompt и service facts]
        Policy[Классификация, эскалация, tool calling]
        CacheAside["Cache-Aside перед LLM<br/>TTL: 3600s<br/>key: sha256(model+messages+temperature+prompt_version+service_facts_version)"]
        Cost[Метрики usage, cost, latency]
    end

    subgraph L["LLM layer"]
        Router[Provider router / fallback chain]
        CB1[Circuit Breaker: OpenAI primary]
        OpenAI[[External: OpenAI-compatible API]]
        CB2[Circuit Breaker: OpenRouter secondary]
        OpenRouter[[External: OpenRouter-compatible API]]
        CB3[Circuit Breaker: Ollama local fallback]
        Ollama[[Local external service: Ollama]]
        Template[Graceful degradation: шаблон + эскалация]
    end

    subgraph D["Data layer"]
        Cache[(Redis или in-memory LLM cache)]
        KB[(Knowledge base JSON сейчас / vector DB позже)]
        Logs[(JSONL logs, usage, feedback)]
        History[(будущее: persistent chat history)]
    end

    Gateway -->|2. нормализованный запрос| Orchestrator
    Orchestrator --> Prompt
    Orchestrator --> Policy
    Policy -->|3. search tool| KB
    Orchestrator -->|4. cache lookup| CacheAside
    CacheAside -->|hit: вернуть cached draft| Orchestrator
    CacheAside -->|miss| Router
    Router --> CB1 --> OpenAI
    CB1 -. provider unavailable .-> CB2
    CB2 --> OpenRouter
    CB2 -. provider unavailable .-> CB3
    CB3 --> Ollama
    CB3 -. unavailable или bad quality .-> Template
    Router -->|5. финальный черновик| Orchestrator
    Orchestrator -->|6. сохранить ответ с TTL| Cache
    Orchestrator -->|7. telemetry| Logs
    Orchestrator -->|8. ответ| Gateway
```

Внешние сервисы на схеме отмечены двойной рамкой: OpenAI-compatible API,
OpenRouter-compatible API и Ollama. Redis и будущие хранилища тоже могут быть
внешними сервисами при deployment, но MVP сохраняет возможность offline-тестов
через in-memory cache и локальные JSONL-логи.

## Cache-Aside

За паттерн Cache-Aside отвечает Service layer. Gateway не обращается к кешу
напрямую, а LLM layer не принимает решение о том, можно ли кешировать ответ.

Проверка кеша выполняется до любого платного вызова LLM-провайдера. Если найден
cache hit, сервис возвращает сохраненный черновик. Если произошел cache miss,
сервис вызывает LLM router, валидирует ответ и только потом записывает значение
в кеш.

Состав ключа кеша:

```text
sha256(model + messages + temperature + prompt_version + service_facts_version)
```

TTL по умолчанию: `3600` секунд для обычных FAQ и ответов по документации.

Не кешируем ответы с высокой temperature, персональные данные, инциденты,
ответы с эскалацией и ошибки провайдеров. Целевой cache hit rate для MVP:
`35-45%`.

## ADR-001: паттерн взаимодействия

### Context

Support Copilot сейчас работает как интерактивный CLI, а позже должен стать
небольшим API-сервисом. Большинство запросов короткие: установка, лицензирование,
типовые ошибки, поиск документации, классификация и определение необходимости
эскалации. Инженеру нужен ответ во время обработки обращения, поэтому задержка
важнее максимальной пропускной способности. При этом система должна оставаться
достаточно простой для учебных модулей: prompts, retry/fallback, cache, function
calling, usage tracking и будущие мультимодальные сценарии должны быть видны в
коде, а не скрыты за избыточной инфраструктурой.

### Decision

Выбранный паттерн взаимодействия: **синхронный Request-Response** для основного
support flow. Gateway принимает одно пользовательское сообщение, Service
собирает prompt и tool context, Cache-Aside проверяет наличие переиспользуемого
ответа, LLM layer генерирует черновик, после чего Gateway возвращает результат
инженеру. Streaming остается опциональным вариантом того же паттерна для длинных
ответов и более живого UX. Queue-based processing не входит в MVP chat path.

### Consequences

Архитектура остается простой для локальной отладки и тестирования. Unit-тесты
могут мокать LLM-клиентов и проверять retry, fallback, cache, tool calls и
эскалацию без брокера сообщений. Минус подхода: каждый пользовательский запрос
занимает worker до завершения или ошибки LLM, поэтому gateway timeouts и
provider timeouts должны быть явными. При росте FastAPI сможет использовать ту
же сервисную логику и добавить request id, auth, rate limits и streaming
responses.

### Alternatives

Event-driven queue рассматривалась для всех LLM-вызовов, но она добавляет
операционную сложность и ухудшает интерактивный support-сценарий. Этот подход
лучше оставить для batch evals, индексации документов, аналитики feedback и
медленных мультимодальных задач, а не для первого chat loop.

## ADR-002: отказоустойчивость

### Context

Ассистент зависит от внешних LLM-провайдеров. Они могут возвращать `429`, `5xx`,
timeouts, невалидные ответы или временные всплески latency. Локальная Ollama
полезна как fallback, но на текущем ноутбуке может отсутствовать или работать
слишком медленно на крупных моделях.

### Decision

Основной провайдер: OpenAI-compatible cloud model из `.env`.

Fallback chain:

```text
OpenAI primary -> OpenRouter/OpenAI-compatible secondary -> local Ollama -> template + escalation
```

У каждого провайдера свой Circuit Breaker. Retry разрешен только для `429`,
`5xx` и timeout с exponential backoff + jitter. Не повторяем `400`, `401` и
`403`, потому что это ошибки конфигурации, авторизации или контракта запроса.
Если все провайдеры недоступны, сервис возвращает понятный graceful-degradation
ответ и предлагает эскалацию человеку.

### Consequences

Пользователь получает полезный ответ даже при отказе основного провайдера.
Компромисс: fallback-ответы могут отличаться качеством, latency и стоимостью.
Поэтому каждый ответ провайдера нужно логировать с model, latency, token usage и
fallback reason. Состояние Circuit Breaker стоит показывать в `/stats` или в
будущих observability endpoints.

## Возможные точки отказа

| Слой | Что может сломаться | Паттерн защиты | Graceful degradation |
| --- | --- | --- | --- |
| Gateway | Неверная команда, слишком длинный ввод, disconnect клиента, будущая ошибка auth/rate-limit | Input validation, size limit, request id, понятные ошибки, timeout | Попросить сократить запрос или повторить; CLI продолжает работать |
| Service | Потерян prompt-файл, несовпадение tool schema, ошибка классификатора, плохой JSON от модели, prompt injection | Версионированные prompts, строгие output contracts, schema validation, injection rules, unit tests | Вернуть безопасный support-ответ; эскалировать при неуверенной классификации |
| LLM | `429`, `5xx`, timeout, outage провайдера, переполнение context window, слабое качество fallback | Retry with backoff, Circuit Breaker на провайдера, fallback chain, context window trimming | Переключиться на другого провайдера; если все отказали, вернуть шаблон и рекомендовать эскалацию |
| Data | Недоступен cache, stale cache, не найдена база знаний, ошибка записи лога, будущий DB outage | Cache-aside with bypass, TTL, versioned cache key, optional Redis, local fallback logs | Продолжить без кеша; ответить по prompt/service facts; предупредить, что данные поиска могут быть неполными |

## Нагрузка и бюджет

Начальные assumptions для MVP:

| Метрика | Цель / допущение |
| --- | --- |
| Активные инженеры поддержки | 5 |
| Запросов на инженера в день | 60 |
| Среднее число запросов в день | 300 |
| Peak RPM | 30 сейчас, до 50 после API rollout |
| Средний input size | 900 tokens |
| Средний answer size | 250-450 tokens |
| Средний total tokens per request | 1,250 tokens |
| Peak TPM при 30 RPM | около 37,500 TPM |
| Daily token volume before cache | около 375,000 tokens |
| Target cache hit rate | 35-45% |
| Daily token volume after 40% cache hit | около 225,000 billable tokens |
| MVP budget guardrail | не больше 5 USD/day для обычного support-трафика |
| Latency target | первый полезный текст до 1.5 секунды в streaming real mode; полный ответ до 8 секунд для обычных запросов |

Точные model id, цены и лимиты намеренно не захардкожены в этом документе.
Перед production-выбором модели нужно проверить актуальную документацию
провайдера и обновить `.env.example`, а не бизнес-логику.

## LiteLLM

LiteLLM рассматривается как кандидат на LLM gateway, но не является обязательной
runtime-зависимостью для текущего CLI. По официальной документации LiteLLM Proxy
настраивается через `config.yaml`, может читать секреты из environment
variables, предоставляет OpenAI-compatible API и поддерживает routing strategies,
retries, fallbacks, timeouts и cooldowns.

Подготовленный конфиг:

```text
docs/litellm/config.yaml
```

Команды для локального эксперимента:

```powershell
python -m pip install "litellm[proxy]"
litellm --config docs/litellm/config.yaml --detailed_debug
```

Пример запроса:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:4000/v1/chat/completions" `
  -Method Post `
  -Headers @{ Authorization = "Bearer $env:LITELLM_MASTER_KEY" } `
  -ContentType "application/json" `
  -Body '{"model":"support-copilot-primary","messages":[{"role":"user","content":"Как исправить ошибку лицензии?"}]}'
```

Симуляция отказа primary provider:

```powershell
$env:OPENAI_API_KEY = "invalid-key-for-fallback-test"
litellm --config docs/litellm/config.yaml --detailed_debug
```

После этого нужно отправить тот же запрос и проверить, что proxy пытается пройти
по fallback chain. Для полноценного успешного fallback-теста secondary provider
или local Ollama должны быть настроены валидными credentials/runtime.

Решение для текущего MVP: **пишем и оставляем собственный lightweight
OpenAI-compatible client** в `support_copilot/infrastructure`, а CLI пока не
проводим через LiteLLM. Причина: задания курса требуют явной локальной
реализации retry, fallback, cache, usage tracking и Function Calling. LiteLLM
становится полезным, когда проект переходит от одного CLI к общему API gateway
для нескольких клиентов, команд, провайдеров, virtual keys, budgets и admin
observability.

Источники для раздела LiteLLM:

- https://docs.litellm.ai/docs/proxy/configs
- https://docs.litellm.ai/docs/routing
- https://github.com/BerriAI/litellm
