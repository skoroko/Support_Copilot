# Технологии Support Copilot

Срез реализации — 21.10.2026. Документ описывает принятый Telegram-контур,
вспомогательные API и учебные эксперименты с явным разделением их ролей.
Численные результаты находятся в [протоколе приёмки](final_acceptance.md),
настройки принятого запуска — в [техническом описании](final_technical_documentation.md).
Точные версии Python-пакетов фиксирует [uv.lock](../uv.lock), диапазоны и
optional extras — [pyproject.toml](../pyproject.toml). Наличие библиотеки
в зависимостях само по себе не означает её использование в каждом запросе.

## Приложение и взаимодействие

| Технология | Для чего используется | Реализация |
|---|---|---|
| Python 3.13, uv | Единое окружение разработки и сборки; диапазон Python `>=3.13,<3.14`, установка из lock-файла | [pyproject.toml](../pyproject.toml), [Dockerfile](../Dockerfile) |
| FastAPI, Pydantic v2, pydantic-settings | HTTP-маршруты, проверка схем, зависимости, настройки и lifecycle приложения | [main.py](../app/main.py), [chat routes](../app/chat/routes.py), [config.py](../app/core/config.py) |
| Uvicorn | ASGI-сервер backend и внутреннего API уведомлений бота | [Compose](../compose.yaml), [bot/web.py](../bot/web.py) |
| aiogram | Команды Telegram, меню, кнопки, вложения и отображение ответа | [bot](../bot/) |
| HTTPX | Асинхронное обращение бота к backend, connection pool, таймауты | [BackendClient](../bot/services/backend_client.py) |
| Starlette StreamingResponse, SSE | Поток событий ответа, метаданных, источников, согласий и ошибок; heartbeat поддерживает соединение | [chat routes](../app/chat/routes.py), [sse.py](../app/core/sse.py) |

Бот — адаптер интерфейса. Он обращается к `/chats`, а бизнес-сценарий выполняют
`ChatService`, `ChatAgent` и persistent graph. В принятом агенте provider stream
сначала собирается внутри `AgentModel`; итог проходит grounding и модерацию.
SSE и renderer поддерживают потоковый протокол, но это не означает немедленный
показ сырых токенов модели в Telegram.

## Модель и исполнение агента

| Технология | Роль | Реализация |
|---|---|---|
| OpenAI SDK | Граница внешнего LLM API; также embeddings, транскрипция и опциональная облачная модерация | [runtime_model.py](../app/services/runtime_model.py), [embeddings.py](../app/services/embeddings.py), [media.py](../app/chat/media.py) |
| LangChain Core / OpenAI | `AIMessage`, сообщения, схемы инструментов и совместимый model adapter | [runtime_model.py](../app/services/runtime_model.py) |
| LangGraph | Явные переходы model/tool loop, сохранение состояния, interrupt и resume | [agent_persistent.py](../app/services/agent_persistent.py) |
| tiktoken | Оценка размера контекста, предварительный прогрев tokenizer до готовности приложения | [ChatService](../app/chat/service.py), [main.py](../app/main.py) |

Принятая модель генерации — `gpt-6-luna`, Responses API, reasoning `medium`,
service tier `priority`; backup generation выключена. ID и параметры задаются
конфигурацией. Краткий и подробный ответы имеют видимые цели 800 и 1500 токенов;
адаптер добавляет резерв reasoning 8192 к общему provider output cap. API не
предоставляет здесь отдельного жёсткого потолка видимого текста. Это ограничение
выхода, а не денежный бюджет всего запроса.

`support_copilot/` сохраняет ранний CLI и самостоятельный клиент с retry,
fallback и кешем. Учебный `agent_graph.py` сравнивает графы с имитацией Telegram.
Они не являются промежуточными слоями текущего Telegram-запроса. Исторический
конфиг LiteLLM не означает наличие proxy в принятом запуске.

## Загрузка документов и поиск

| Технология | Роль | Реализация |
|---|---|---|
| LlamaIndex | Документы и nodes, ingestion pipeline, docstore UPSERTS и интеграция vector store | [ingestion.py](../app/services/ingestion.py) |
| Qdrant 1.19.0 | Dense-векторы и metadata-фильтры; принятый индекс — Cosine, 1024 измерения | [rag.py](../app/services/rag.py), [Compose](../compose.yaml) |
| OpenAI embeddings через LlamaIndex adapter | Принятый retrieval/ingestion: `text-embedding-3-large`, 1024 измерения | [retrieval_embedding.py](../app/services/retrieval_embedding.py), [rag.py](../app/services/rag.py) |
| Собственный BM25 и RRF | Лексический поиск, объединение независимых выдач, дедупликация | [lexical.py](../app/services/lexical.py), [rag.py](../app/services/rag.py) |
| pdfplumber, pypdf | Структурное извлечение PDF, страницы и таблицы; fallback для текста | [structured_readers.py](../app/services/structured_readers.py), [media.py](../app/chat/media.py) |
| BeautifulSoup, python-docx, pylibmspack | Чтение HTML, DOCX и CHM соответственно | [ingestion.py](../app/services/ingestion.py), [chm_reader.py](../app/services/chm_reader.py) |
| Pillow | Декодирование и проверка размера загруженных изображений | [media.py](../app/chat/media.py) |

Корпус принятой версии состоит из девяти PDF и одного CHM по GTS NX и Civil.
Поддержка загрузчиками других форматов не означает их присутствия в корпусе.
Производные HTML-зеркала и синтетический FAQ исключены. Полный состав,
извлекаемость и пробелы описаны в [инвентаризации](data_inventory.md).

Структурный `EvidenceSplitter` применяет recursive-разбиение с настройками
1024/64, сохраняя атомарность строк таблиц и другого структурного evidence.
Фиксированное и semantic-разбиение оставлены для экспериментов. PDF extraction
не является OCR; изображения и формулы без проверенной транскрипции остаются
явным ограничением. Сопутствующие отношения и проверенные транскрипции
привязаны к идентификатору и хэшу исходника.

## Хранилища и согласованность

| Технология | Что хранит | Граница |
|---|---|---|
| PostgreSQL 16, SQLAlchemy async, asyncpg, Alembic | Чаты, сообщения и прикладные записи; миграции схемы | [chat repository](../app/chat/repository.py), [migrations](../alembic/) |
| psycopg pool, LangGraph Postgres saver | Состояние выполнения и checkpoints графа | [agent_persistent.py](../app/services/agent_persistent.py) |
| Redis 7.4 | TTL-кеш общего LLM API и состояние дедупликации уведомлений | [llm.py](../app/services/llm.py), [bot/web.py](../bot/web.py) |
| SQLite / aiosqlite | Локальный вариант checkpoints; отдельно embedding cache учебных скриптов, не RAG adapter | [agent_persistent.py](../app/services/agent_persistent.py), [embeddings.py](../app/services/embeddings.py) |
| JSON / JSONL, filelock | Локальная реализация chat repository | [chat repository](../app/chat/repository.py) |
| JSON docstore и каталог корпуса | Учёт загруженных ревизий и структурного происхождения evidence | [ingestion.py](../app/services/ingestion.py), [corpus.py](../app/services/corpus.py) |

В Compose история и checkpoints используют PostgreSQL, но сохраняются разными
механизмами и транзакциями. Единой атомарной транзакции «ответ + checkpoint +
доставка Telegram» нет. Блокировки чата и admission quotas действуют внутри
одного API-процесса; принятая конфигурация использует один worker.

## Проверки, наблюдаемость и поставка

| Инструмент | Назначение и статус |
|---|---|
| pytest / pytest-asyncio, Ruff | Offline-проверки и статический анализ; результаты сдаваемой версии отдельно зафиксированы в протоколе |
| RAGAS, pandas, rapidfuzz и собственные eval runners | Учебная и регрессионная оценка retrieval, generation, цитат и latency; не выполняются на каждом пользовательском запросе |
| structlog | JSON-логи с request context, latency и безопасными метаданными; маскирование PII |
| Phoenix, OpenTelemetry, OpenInference | Опциональные tracing extra и Compose profile `observability`; содержимое сообщений, документов и embeddings скрывается |
| sentence-transformers | Опциональный локальный BGE reranker; в принятом профиле reranker выключен |
| Docker / Docker Compose | Multi-stage сборка с locked uv, прогрев tokenizer, непривилегированный пользователь; app, bot, PostgreSQL, Redis, Qdrant и optional Phoenix с volumes и readiness |

Меры защиты реализованы кодом приложения: служебная аутентификация, ownership,
ограничения входа и вложений, локальные правила модерации, изоляция парсинга,
разделение tools по полномочиям и подтверждение внешней отправки. Опциональная
облачная модерация при недоступности допускает продолжение после локальных
проверок. Ни одна из этих мер не является доказательством полной защиты от
prompt injection. Обоснование приёмов и компромиссы:
[архитектурные решения](architecture_patterns.md).
