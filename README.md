# Support Copilot

## Тема

ИИ-ассистент техподдержки.

## Описание проекта

Support Copilot — внутренний ИИ-ассистент для технических специалистов компании MIDAS. Проект поможет команде поддержки быстрее находить ответы на типовые вопросы пользователей, готовить черновики ответов и работать с базой знаний, документацией, историей обращений и внутренними инструкциями.

Ассистент не заменяет инженера поддержки, а выступает как инструмент подсказок: предлагает возможное решение, показывает релевантные материалы и помогает понять, когда обращение нужно передать специалисту более высокого уровня.

## Почему выбрана эта тема

Тема выбрана потому, что техническая поддержка инженерного программного обеспечения часто сталкивается с повторяющимися вопросами: установка, лицензирование, типовые ошибки, поиск нужного раздела документации, базовые методические вопросы. Такие обращения занимают время специалистов первой линии и отвлекают продуктовые команды.

ИИ-ассистент позволит снизить нагрузку на поддержку, ускорить обработку обращений и сделать знания компании более доступными для сотрудников.

## Основной запуск

Рабочий CLI вынесен в пакет `support_copilot`, чтобы следующие домашние задания
можно было подключать к одному приложению, а папки `homework_*` оставались
учебными wrappers и demo-материалами.

Настройки провайдеров, API-ключи, модели, цены и таймауты читаются из `.env`
через `support_copilot/config.py` на базе `pydantic-settings`. Секреты не
хранятся в коде; `.env.example` содержит только имена переменных.

Из папки `HelpDesk_Assistant`:

```powershell
python -m support_copilot
```

Real mode:

```powershell
python -m support_copilot --real
```

Function Calling mode с доменным tool `search_knowledge_base`:

```powershell
python -m support_copilot --tools
```

Real Function Calling mode:

```powershell
python -m support_copilot --real --tools
```

## Какую задачу решает проект

Проект решает задачу внутренней помощи техническим специалистам при обработке обращений пользователей. В рамках будущего MVP планируется реализовать ассистента, который:

- ищет ответы в базе знаний и документации;
- предлагает черновик ответа инженеру поддержки;
- помогает классифицировать тип обращения;
- подсказывает, когда вопрос нужно эскалировать;
- собирает обратную связь о полезности ответа.

## Домашнее задание 1.3: локальные модели Ollama

Для первого локального инкремента выбран маленький основной кандидат
`qwen3:1.7b`: он легкий, подходит для ноутбука с 32 ГБ RAM и лучше ожидаемо
поддерживает русский язык, чем многие компактные англоязычные модели.

Материалы ДЗ находятся в `homework_1_3/`:

- `benchmark_ollama.py` — benchmark 3 локальных моделей на задачах техподдержки;
- `Modelfile` — кастомная модель `support-copilot-qwen3-1.7b`;
- `README.md` — команды установки, запуска и описание результата.

## Домашнее задание 1.4: сравнение моделей и стоимость

Для полного цикла выбора модели подготовлен eval-набор из 21 support-сценария,
benchmark локальных и облачных моделей, а также расчет стоимости OpenAI API.

Основной локальный кандидат: `qwen3:1.7b`. Облачный кандидат задается через
`OPENAI_MODEL` в `.env`; конкретный model id нужно выбрать по актуальной
документации провайдера.

Материалы ДЗ находятся в `homework_1_4/`:

- `eval_set.json` — 21 задача с проверочными ключевыми словами;
- `benchmark_models.py` — benchmark Ollama и OpenAI-compatible API;
- `cost_calculator.py` — расчет месячной стоимости без кэша и с кэшем;
- `README.md` — команды запуска, assumptions и рекомендация.

## Домашнее задание 2.1: API провайдеров

Подготовлен демонстрационный OpenAI-compatible API client через OpenAI SDK:
загрузка `.env`, system/user messages, temperature experiment, streaming,
переключение `openai`/`ollama` через `LLM_PROVIDER`, замер TTFT и полного
времени ответа.

Материалы ДЗ находятся в `homework_2_1/`:

- `provider_api_demo.py` — Python-скрипт для демонстрации API провайдеров;
- `README.md` — команды запуска и комментарии по temperature;

Файлы окружения:

- `.env.example` — шаблон переменных без секретов;
- `.env` — локальный файл для ключей, не должен попадать в git.

## Домашнее задание 2.2: промпт-инжиниринг

Подготовлен production-ready system prompt для Support Copilot по структуре
РРФО, few-shot примеры, tiktoken-проверка длины prompt, автотесты
`should_contain` / `should_not_contain` и сравнение ответов с few-shot и без.

Материалы ДЗ находятся в `homework_2_2/`:

- `prompt_engineering_demo.py` — Python-файл с prompt, тестами и демонстрацией;
- `README.md` — команды запуска и описание проверок;
- `results/test_run_log.txt` — лог проверенного запуска.

## Домашнее задание 2.3: надежность LLM API

Подготовлен `RobustLLMClient`: OpenAI SDK wrapper с retry для `429`, `5xx` и
timeout, exponential backoff + jitter, fallback `OpenAI -> OpenRouter -> Ollama -> служебное сообщение`, логированием, usage/cost tracking, circuit
breaker и client-side rate limiter.

Материалы ДЗ находятся в `homework_2_3/`:

- `robust_llm_client.py` — класс надежного клиента;
- `demo_retry_fallback.py` — offline-демонстрация retry и fallback;
- `README.md` — команды запуска и настройки `.env`;
- `results/retry_fallback_log.txt` — лог/описание проверенного сценария.

## Домашнее задание 2.4: кеширование LLM

Подготовлен `LLMCache` для in-memory кеширования ответов LLM с SHA-256 ключом,
TTL, статистикой hits/misses и hit rate. Также добавлен `RedisLLMCache` с тем же
интерфейсом, JSON-сериализацией и TTL через Redis.

Материалы ДЗ находятся в `homework_2_4/`:

- `llm_cache.py` — `LLMCache` и `RedisLLMCache`;
- `demo_cache.py` — демонстрация miss/hit/TTL/stats;
- `README.md` — команды запуска in-memory и Redis demo;
- `results/cache_demo_log.txt` — лог/описание проверенного сценария.

## Домашнее задание 2.5: CLI-помощник техподдержки

Собран интерактивный `Support Copilot CLI`, объединяющий блоки 2.1-2.4:
system prompt и few-shot, retry/fallback, кеш, команды `/clear`, `/stats`,
`/quit`, историю последних 10 сообщений, классификацию, эскалацию, streaming
через `--stream` и логирование в файл.

Основной код находится в `support_copilot/`, а материалы ДЗ находятся в
`homework_2_5/`:

- `support_cli.py` — совместимый wrapper на `support_copilot.cli`;
- `README.md` — команды запуска offline demo и real-provider режима;
- `results/session_demo_log.txt` — лог scripted demo с 5+ сообщениями.

## Домашнее задание 2.6A: Vision API

Добавлен мультимодальный сценарий анализа изображений в общий CLI: команда
`/image <путь> | <вопрос>` принимает пользовательский PNG/JPEG-файл, кодирует
изображение в base64 data URL и в real mode отправляет его в OpenAI-compatible
Vision API. Отдельный `vision_assistant.py` оставлен как helper/smoke-test, но
основной пользовательский сценарий теперь работает через `python -m support_copilot`.

Материалы ДЗ находятся в `homework_2_6/`:

- `vision_assistant.py` — helper для анализа изображения через Vision API;
- `README.md` — команды запуска и настройки `.env`;
- основной запуск: `python -m support_copilot --real`, затем команда `/image`.

## Домашнее задание 3.1: Function Calling

Добавлен доменный tool `search_knowledge_base(query, product)` для поиска по
локальной базе знаний Support Copilot. Реализован полный цикл OpenAI-compatible
Function Calling: JSON Schema, model request с `tools`, парсинг `tool_calls`,
выполнение Python-handler, возврат `role: tool` в историю и второй запрос за
финальным ответом.
Real-provider настройки для этого режима также берутся из `support_copilot/config.py`.

Основной CLI тоже умеет запускать этот режим:

```powershell
python -m support_copilot --tools
```

Материалы ДЗ находятся в `homework_3_1/`:

- `run_tool_call_demo.py` — offline demo и real mode через `.env`;
- `README.md` — описание tool, запуск и 3 наблюдения;
- `results/tool_call_demo.jsonl` — структурированный лог последнего demo-запуска;
- `results/demo_summary.md` — краткая таблица результатов.

## Домашнее задание 3.2: архитектура ИИ-приложения

Подготовлен архитектурный паспорт проекта для следующих модулей: 4 слоя
Gateway / Service / LLM / Data, request flow, Cache-Aside перед LLM, Circuit
Breaker на провайдерах, fallback chain, ADR по паттерну взаимодействия, отказоустойчивость,
таблица failure points, нагрузочные assumptions и решение по LiteLLM.

Материалы:

- `docs/architecture.md` — основной архитектурный паспорт и ADR;
- `docs/litellm/config.yaml` — пример LiteLLM Proxy config без секретов.

## Домашнее задание 3.3: async LLM и SSE

Добавлен асинхронный LLM-клиент поверх `AsyncOpenAI`: `complete`,
`batch_chat`, `stream_chat`, instance-level `Semaphore`, SDK retry
`max_retries=3`, timeout бизнес-логики и per-request logging. Также добавлен
FastAPI SSE endpoint `/chat/stream` и benchmark sync sequential vs async batch.

Команды:

```powershell
python scripts/benchmark.py
uvicorn support_copilot.api:app --reload --port 8000
curl -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d "{\"prompt\":\"Что такое event loop?\"}"
```

Материалы:

- `homework_3_3/README.md` — краткая инструкция для проверки ДЗ;
- `support_copilot/infrastructure/async_llm.py` — `AsyncLLMClient`;
- `support_copilot/api.py` — FastAPI + SSE;
- `scripts/benchmark.py` — benchmark;
- `scripts/benchmark_results.md` — результат локального demo-бенчмарка.

## Домашнее задание 3.4: FastAPI LLM Service

Собран отдельный FastAPI-сервис `app.main:app` с dependency injection, lifespan
инициализацией `AsyncOpenAI` и Redis-compatible cache, middleware для
`X-Request-ID`, обработчиками ошибок, OpenAPI examples/responses и четырьмя
эндпоинтами: `/health`, `/models`, `/chat`, `/chat/stream`.

Запуск из папки `HelpDesk_Assistant`:

```powershell
uvicorn app.main:app --reload --port 8000
```

Проверка без LLM-вызова:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/models
```

Проверка chat API при настроенном `OPENAI_API_KEY`:

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Как проверить срок действия лицензии?\"}],\"temperature\":0.2,\"max_tokens\":500}"
curl -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Сформулируй короткий ответ клиенту.\"}],\"temperature\":0.2,\"max_tokens\":500}"
```

Материалы:

- `homework_3_4/README.md` — инструкция запуска и проверки;
- `homework_3_4/TEACHER_CHECK.md` — краткая сводка для преподавателя;
- `app/main.py` — FastAPI app, lifespan, middleware, exception handlers;
- `app/services/llm.py` — `LLMService` с complete, stream и cache-aside;
- `app/deps/providers.py` — FastAPI DI providers без глобальных клиентов;
- `tests/test_app_fastapi.py`, `tests/test_app_llm_service.py` — offline-тесты.

## Домашнее задание 3.5: Docker и контейнеризация

FastAPI-сервис из 3.4 упакован в Docker: добавлен multi-stage `Dockerfile` на
`python:3.13-slim-bookworm` с установкой зависимостей через `uv`, runtime-слоем
без root-пользователя и запуском `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
Также добавлен `compose.yaml` со связкой `app + redis`, healthcheck-ами,
`depends_on: service_healthy` и named volume для Redis.

Запуск стека:

```powershell
Copy-Item .env.example .env
# заполнить OPENAI_API_KEY в .env
docker compose up -d --build
```

Проверка:

```powershell
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/ready
docker compose exec app id
docker compose exec redis redis-cli ping
```

Материалы:

- `Dockerfile` — multi-stage image, `uv`, non-root `appuser`, exec-form CMD;
- `.dockerignore` — исключает `.env`, `.git`, tests, cache и лишние артефакты;
- `compose.yaml` — сервисы `app` и `redis`, healthcheck-и и volume;
- `app/routers/health.py` — liveness `/health` и readiness `/ready`;
- `homework_3_5/TEACHER_CHECK.md` — краткая сводка для преподавателя;
- `tests/test_docker_config.py` — offline-проверка Docker/Compose контракта.

## Домашнее задание 3.6: Observability ИИ-приложений

К FastAPI-сервису добавлен observability-слой: Phoenix в Docker Compose,
OpenAI auto-instrumentation через OpenInference, JSON-логи через `structlog`,
`request_id` в contextvars и PII-маскирование prompt preview перед записью в лог.

Запуск:

```powershell
docker compose up -d --build
```

Проверка:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:6006
docker compose logs app
```

Материалы:

- `app/observability/tracing.py` — Phoenix/OpenInference setup;
- `app/observability/logging.py` — structlog JSON logging и contextvars;
- `app/observability/pii.py` — regex-маскирование EMAIL, PHONE_RU, CARD, INN, PASSPORT;
- `compose.yaml` — сервис `phoenix`, volume `phoenix-data`, endpoint для `app`;
- `docs/observability/README.md` — описание того, что видно в Phoenix;
- `homework_3_6/TEACHER_CHECK.md` — краткая сводка для преподавателя;
- `tests/test_observability.py` — unit-тесты PII и безопасного LLM-лога.

## Домашнее задание 3.7: тестирование и оценка качества ИИ-приложений

К FastAPI-сервису добавлен отдельный evaluation-слой рядом с `tests/`: golden
dataset на 22 Support Copilot кейса, быстрые unit-тесты вокруг LLM-логики,
ручной LLM-as-judge runner и threshold checker для решения "можно/нельзя
релизить" после смены prompt или модели.

Быстрые проверки без сети и ключей:

```powershell
python -m pytest tests/unit/ -m "not llm"
python eval/run_evaluation.py --offline-smoke --out eval/runs/2026-07-07.json
python eval/check_thresholds.py eval/runs/2026-07-07.json
```

Реальный evaluation-прогон:

```powershell
uvicorn app.main:app --port 8000
python eval/run_evaluation.py --golden eval/golden_dataset.json --judge gpt-5.2 --out eval/runs/2026-07-07.json
```

Материалы:

- `eval/golden_dataset.json` — versioned golden dataset;
- `eval/run_evaluation.py` — G-Eval style reason-then-score runner;
- `eval/check_thresholds.py`, `eval/thresholds.yaml` — проверка quality gates;
- `tests/unit/` — unit-тесты prompt builder, parsers, schemas, cost, cache, retry;
- `homework_3_7/TEACHER_CHECK.md` — краткая сводка для преподавателя.

## Домашнее задание 3.8: безопасность ИИ-приложений

К FastAPI LLM Service добавлен защитный слой для red-team проверок: входной
валидатор prompt injection/encoding attacks, canary-token в системном сообщении,
output filter от утечки system prompt/canary и небезопасного Markdown/HTML, а также
PII-маскирование исходящих ответов и `answer_preview` в LLM-логах.

Быстрые проверки:

```powershell
python -m pytest
python -m compileall app eval scripts homework_3_8 tests
```

Garak REST config:

```powershell
uvicorn app.main:app --port 8000
garak --target_type rest -G eval/security/rest_config.json --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 --generations 1 --report_prefix after
```

Материалы:

- `app/services/security/input_validator.py` — проверка входа до LLM;
- `app/services/security/output_filter.py` — canary, prompt leakage и PII filter;
- `eval/security/rest_config.json` — REST target для garak;
- `docs/security/garak_baseline_2026-07-07.md`, `docs/security/garak_after_2026-07-07.md` — шаблоны отчётов baseline/after;
- `homework_3_8/TEACHER_CHECK.md` — краткая сводка и команды проверки.

## Домашнее задание 4.3: мультимодальность и streaming

Связка FastAPI backend + Telegram bot расширена единым multipart endpoint для
текста, фото, голоса и PDF/DOCX. Изображения передаются в основной multimodal
chat-вызов, аудио транскрибируется без FFmpeg, документы извлекаются на backend,
а готовый `media_refs.part` сохраняется в истории. Бот использует нативный
`sendMessageDraft`, singleton `httpx.AsyncClient` и защищённый `/notify` для
проактивных сообщений.

Материалы и команды проверки: `homework_4_3/README.md` и
`homework_4_3/TEACHER_CHECK.md`.

## Домашнее задание 4.4: production-обвязка

Backend расширен двухслойной модерацией, защищёнными admin endpoints,
24-часовой статистикой, feedback-хранилищем и очередью Telegram-рассылок.
Telegram-бот получил router-level проверку `BOT_ADMIN_IDS`, команды `/stats`,
`/users`, `/broadcast`, кнопки 👍/👎 и фоновую обработку очереди. Compose запускает
`app + bot + postgres` одной командой, применяет миграции и сохраняет PostgreSQL
в named volume `pg-data`.

Материалы и команды проверки: `homework_4_4/README.md` и
`homework_4_4/TEACHER_CHECK.md`.

## Домашнее задание 5.1: эмбеддинги и семантический поиск

Добавлен переиспользуемый OpenAI embedding-сервис с публичным `embed_texts`,
батчингом, L2-нормализацией, точной проверкой token limit, retry только для
временных ошибок и постоянным SQLite-кешем, который инвалидируется при смене
модели или размерности. Mini-benchmark содержит 8 предметных русскоязычных пар.
Выбор `text-embedding-3-large` с 1024 dimensions и расчёт стоимости индексации
60 документов описаны в `homework_5_1/README.md`; краткая проверка - в
`homework_5_1/TEACHER_CHECK.md`.

## Домашнее задание 5.2: векторное хранилище Qdrant

В Compose добавлен Qdrant 1.19 с API key, named volume и healthcheck. Приложение
использует один `AsyncQdrantClient` из lifespan, проверяет коллекцию размерности
1024 и создаёт HNSW/payload indexes. Идемпотентный загрузчик индексирует 120
учебных фрагментов Support Copilot с детерминированными UUID, а отдельный
эксперимент сравнивает Cosine/Dot и проверяет строковый, datetime и составной
payload-фильтры.

Материалы и команды проверки: `homework_5_2/README.md`,
`homework_5_2/TEACHER_CHECK.md` и `docs/vector_store.md`.

## Домашнее задание 5.3: RAG с LlamaIndex и без фреймворка

Добавлен двуязычный корпус из десяти руководств GTS NX/Civil, включая отдельный
загрузчик русской CHM-справки. Основная реализация использует LlamaIndex и
коллекцию `rag_block_03`, а ручной baseline явно выполняет загрузку, chunking,
embedding, `query_points` и Chat Completions в коллекции
`rag_block_03_baremetal`. Endpoint `POST /rag/query` возвращает ответ, top score
и три источника. При `RAG_ENABLED=true` приложение по умолчанию выполняет
инкрементальный UPSERT при старте; для уже подготовленной коллекции можно
указать `RAG_INGEST_ON_START=false` и подключиться к ней без переиндексации.

Материалы и команды проверки: `homework_5_3/README.md`,
`homework_5_3/TEACHER_CHECK.md`, `homework_5_3/questions.json` и `docs/rag.md`.

## Домашнее задание 5.4: chunking и оптимизация retrieval

Для десяти документов подготовлен golden dataset на 28 вопросов и реализовано
сравнение `fixed`, `recursive`, `semantic` chunking в отдельных коллекциях
Qdrant. Эксперимент считает Hit Rate@5, MRR@10, Recall@10, статистику chunks и
latency, сравнивает локальный `BAAI/bge-reranker-v2-m3` до/после и проверяет
четыре конфигурации `chunk_size/overlap/top-K`.

По локальному воспроизводимому прогону выбрана конфигурация
`recursive/512/64/top-K=10`: Hit Rate@5 `0.929`, MRR@10 `0.771`, Recall@10
`0.750`. BGE повысил MRR@10 до `0.839`, но CPU latency сделала его опциональным
этапом.

Материалы и команды проверки: `homework_5_4/README.md`,
`homework_5_4/TEACHER_CHECK.md`, `tests/eval/retrieval_dataset.json` и
`docs/chunking_experiment.md`.

## Домашнее задание 5.5: корпоративный RAG-ассистент

RAG собран в единый инкрементальный контур: 70 реальных файлов PDF/HTML/CHM
загружаются через multi-format readers, обогащаются метаданными и индексируются
`IngestionPipeline` с `SimpleDocumentStore` и `DocstoreStrategy.UPSERTS`.
Повторный офлайн-прогон подтверждает идемпотентность: `0 changed, 1992
unchanged`, без дублирования 3508 существующих чанков.

Query pipeline выполняет retrieval `top_k=10`, optional BGE re-ranking до пяти
фрагментов и score guard до вызова LLM. Ответ содержит цитаты `[1]`, `[2]` и
структурированные sources. Postgres-история участвует в multi-turn retrieval,
чат отправляет источники финальным SSE-событием, Telegram обновляет одно
сообщение с debounce 0.8 секунды, а feedback сохраняет показанные источники.
`POST /documents/upload` возвращает 202 и запускает background UPSERT.

Материалы и команды проверки: `homework_5_5/README.md`,
`homework_5_5/TEACHER_CHECK.md`, `docs/rag.md` и `docs/data_inventory.md`.
