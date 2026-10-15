# Support Copilot

Support Copilot — внутренний ИИ-ассистент для инженеров технической поддержки MIDAS.
Основной интерфейс — Telegram: агент ищет сведения в мануалах, отвечает с
источниками, помогает подготовить ответ пользователю и запрашивает подтверждение
перед отправкой сообщения. Решение по обращению остаётся за инженером поддержки.

**Текущий сценарий:** пользователь в Telegram → агент → поиск по мануалам →
ответ с источниками → по просьбе пользователя черновик внешнего сообщения →
подтверждение или отмена отправки.

## Как пользоваться

1. Откройте настроенного Telegram-бота и отправьте `/start`. Меню содержит
   примеры вопросов, инструкцию, историю, ожидающие отправки и новый диалог.
2. Напишите вопрос, указав продукт, версию и детали задачи. Агент использует
   поиск по мануалам; ссылки `[1]`, `[2]` и кнопка «Документы» под ответом
   позволяют проверить найденные источники и доступные номера страниц.
3. Задавайте уточнения в том же диалоге или попросите подготовить черновик
   ответа пользователю. История сохраняется после перезапуска бота.
4. Если нужна отправка, укажите числовой `chat_id` получателя и текст.
   Бот покажет точного адресата и полный текст с кнопками «Подтвердить» и
   «Отменить». Отправка выполняется только после подтверждения этого черновика.

Краткий ответ ограничен 500 выходными токенами по умолчанию. Кнопка «Подробнее»
сначала предупреждает о дополнительном запросе к ИИ и расходе токенов; только
«Подготовить подробный ответ» запускает генерацию с лимитом 1500 токенов по уже
найденным фрагментам. Открытие меню и отмена подробного ответа не вызывают модель.

| Команда / кнопка | Действие |
|---|---|
| `/start` | Открыть меню и восстановить ожидающее подтверждение после перезапуска |
| `/menu`, `/help` | Открыть навигацию или инструкцию |
| `/history` | Показать последние десять сообщений |
| «Ожидающие отправки» | Проверить адресата и текст подготовленного сообщения |
| «Новый диалог» | Очистить историю и ожидающую отправку после отдельного подтверждения |
| `/clear` | Сразу очистить историю и отменить ожидающую отправку |

## Запуск Telegram-ассистента

Все команды ниже выполняются из каталога `HelpDesk_Assistant/`, где находятся
этот README и [compose.yaml](compose.yaml). Для основного запуска нужны Docker
Desktop с Linux-контейнерами и Docker Compose, доступ к OpenAI API и токен
Telegram-бота. Python на хосте требуется только для локальных проверок.

1. Создайте локальный `.env` из шаблона, если файл ещё не существует:

   ```powershell
   if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
   ```

2. Заполните обязательные переменные из таблицы ниже. Включите `RAG_ENABLED=true`.
   Для облачных embeddings явно разрешите передачу содержимого мануалов и
   поисковых запросов провайдеру через `RAG_ALLOW_EXTERNAL_DATA_TRANSFER=true`.
   В шаблоне оба флага выключены. Добавьте свой Telegram user ID в
   `BOT_ALLOWED_USER_IDS`: при пустом списке бот отклоняет все входящие сообщения.
3. Проверьте наличие мануалов в `data/rag-block-03/` и HTML-разделов в
   `data/gts_nx_help_html/`; состав и подготовка описаны в
   [инвентаризации корпуса](docs/data_inventory.md). При первом запуске оставьте
   `RAG_INGEST_ON_START=true`: приложение инкрементально заполнит Qdrant из
   `data/`. Для уже заполненной совместимой коллекции можно выставить `false`.
   Первичная индексация использует внешний embedding API, занимает время и
   расходует токены; её завершение нужно дождаться до проверки поиска.

**Единая команда запуска:**

```powershell
docker compose up -d --build
```

Она поднимает FastAPI, Telegram-бота, PostgreSQL, Redis и Qdrant. Миграции базы
применяются при старте приложения; история диалогов и checkpoints агента
сохраняются в PostgreSQL. Дополнительный профиль для бота не нужен.

Проверка состояния:

```powershell
docker compose ps
Invoke-RestMethod http://localhost:18000/ready
```

В ответе `/ready` проверьте `rag: up`: HTTP 200 при `status: degraded` и
`rag: down` означает, что API работает, но поиск по мануалам недоступен.
После готовности RAG откройте своего бота и отправьте `/start`, затем вопрос
из примеров ниже. API доступен на порту `APP_HOST_PORT` (по умолчанию 18000),
[Swagger UI](http://localhost:18000/docs) — на `/docs`.

После изменения `.env` примените настройки пересозданием контейнеров:

```powershell
docker compose up -d --build --force-recreate
```

## Переменные окружения

Основные настройки для запуска через Compose приведены ниже. Полный шаблон —
[.env.example](.env.example). Секреты храните только в локальном `.env` и не
добавляйте его в Git. `INTERNAL_TOKEN` и `ADMIN_TOKEN` должны быть разными,
каждый — не короче 32 символов, без значений-заглушек.

| Переменная | Что указать / значение по умолчанию | Назначение |
|---|---|---|
| `OPENAI_API_KEY` | Обязательный секрет | Доступ к модели и OpenAI embeddings |
| `BOT_TOKEN` | Обязательный секрет; допустим также алиас `TELEGRAM_BOT_TOKEN` | Токен Telegram-бота |
| `INTERNAL_TOKEN` | Обязательный секрет | Общий сервисный токен backend и бота |
| `ADMIN_TOKEN` | Обязательный отдельный секрет, даже без использования админ-команд | Авторизация административных операций |
| `QDRANT_API_KEY` | Задайте непустой секрет | Общий ключ Qdrant и приложения |
| `BOT_ALLOWED_USER_IDS` | Обязательно задайте проверенные user ID через запятую | Допуск пользователей; пусто — все сообщения отклоняются |
| `BOT_ALLOWED_CHAT_IDS` | Пусто; для групп задайте разрешённые chat ID | Допуск групп дополнительно к списку пользователей |
| `BOT_ADMIN_IDS` | Пусто; при необходимости задайте ID администраторов | Админ-команды; ID также должен быть в `BOT_ALLOWED_USER_IDS` |
| `APP_DEFAULT_MODEL` | Модель из шаблона; legacy alias — `OPENAI_MODEL` | Модель Telegram-агента; `APP_DEFAULT_MODEL` имеет приоритет |
| `RAG_ENABLED` | Для этого сценария `true`; в шаблоне `false` | Включить поиск по мануалам |
| `RAG_ALLOW_EXTERNAL_DATA_TRANSFER` | `true` при согласии на облачную обработку; в шаблоне `false` | Передача документов и поисковых запросов embedding-провайдеру; требуется и с готовым индексом |
| `RAG_INGEST_ON_START` | `true`; `false` для готовой коллекции | Инкрементальная индексация при старте |
| `RAG_COLLECTION` | `rag_block_03` | Коллекция мануалов в Qdrant |
| `RAG_PRODUCT` / `RAG_PRODUCT_SCOPE` | `gts_nx` / `fallback` | Основной продукт и область поиска |
| `RAG_EMBEDDING_MODEL` / `RAG_EMBEDDING_DIMENSIONS` | Значение модели из шаблона / `1024` | Должны соответствовать существующему индексу; смена требует новой индексации |
| `CHAT_RESPONSE_TOKENS` | `500` | Лимит краткого ответа |
| `CHAT_EXPANDED_RESPONSE_TOKENS` | `1500` | Лимит подробного ответа по подтверждению |
| `POSTGRES_USER` / `POSTGRES_DB` / `POSTGRES_PASSWORD` | Пользователь, база и собственный URL-безопасный пароль PostgreSQL | Compose формирует подключения истории и checkpoints из этих значений |
| `APP_HOST_PORT` | `18000` | Порт API на хосте |
| `TRACING_ENABLED` | `false` | Опциональные traces; Phoenix запускается профилем `observability` |

Compose самостоятельно задаёт `BACKEND_URL=http://app:8000`, адреса Redis и
Qdrant, `RAG_DATA_DIR=/app/data`, `CHAT_REPOSITORY=postgres` и
`AGENT_CHECKPOINTER=postgres`. Локальные SQLite/JSON-настройки из шаблона
относятся к запуску вне Compose. `RAG_LLM_MODEL` управляет генерацией обычного
RAG API, а Telegram-агент использует `APP_DEFAULT_MODEL`.

Для `gpt-6-luna` агент автоматически использует Responses API. Reasoning задаётся
через `AGENT_REASONING_EFFORT` (`none`, `low`, `medium`, `high`, `xhigh`, `max`);
`AGENT_EXPANDED_REASONING_EFFORT` позволяет выбрать отдельный режим для «Подробнее».
Если значение не задано, GPT-6 использует `low`, прежние модели сохраняют поведение.
`AGENT_REASONING_TOKEN_RESERVE=8192` добавляется к общему лимиту генерации при
включённом reasoning: отдельного жёсткого лимита для видимого текста API не даёт.
Краткость задаётся также инструкцией 3–5 пунктов. `AGENT_GENERATION_TIMEOUT=120`
задаёт таймаут одного вызова модели. Ответы и tool history отправляются с `store=false`;
зашифрованные reasoning items сохраняются в истории для продолжения tool loop.

Модель обычного RAG API настраивается отдельно: в локальном `.env` и шаблоне
выбраны `RAG_LLM_MODEL=gpt-6-luna`, `RAG_REASONING_EFFORT=medium` и
`RAG_CONDENSE_REASONING_EFFORT=none`. Последняя настройка относится к
переформулированию вопроса по истории диалога. Проверка переключения:
[RAG Luna medium](docs/rag_luna_rollout.md). Старые API `/chat` без инструментов и
резервный chat-путь используют GPT-6 с `reasoning_effort=none`.
При смене модели обновите также `APP_ALLOWED_MODELS`, если включён allowlist.
Методика и результаты выбора режима: [Luna eval](docs/luna_runtime_evaluation.md).
Профиль `compose.luna.yaml` задаёт агенту `medium` и «Подробнее» `medium`.
Обновление уже запущенного стека:
`docker compose -f compose.yaml -f compose.luna.yaml up -d --no-deps --build app`.
15.10.2026 профиль активирован в локальном backend; те же значения сохранены
в локальном `.env`, поэтому обычный Compose также сохраняет Luna medium.
Результаты проверки: [переключение бота](docs/luna_runtime_rollout.md).
При подключении `compose.luna.yaml` его секция `environment` имеет приоритет
над `.env`: чтобы выбрать другую модель, измените профиль либо запускайте
основной `compose.yaml` без него. Профиль не переопределяет настройки RAG.
`.env.example` отражает Luna medium для бота и RAG;
`OPENAI_MODEL` отдельно сохраняет модель прежнего CLI.

## Примеры вопросов и действий

| Сценарий | Сообщение в Telegram |
|---|---|
| Найти раздел | «Найди в руководстве GTS NX описание Mesh Generate и укажи источник». |
| Вопрос по мануалу | «Какие граничные условия описаны в руководстве GTS NX?» |
| Уточнение в том же диалоге | «Поясни второй пункт подробнее и укажи раздел руководства». |
| Черновик | «Подготовь короткий ответ пользователю по найденному фрагменту. Сохрани ссылки на источники». |
| Проверка границ знаний | «Сколько стоит лицензия GTS NX?» — тарифов в текущем корпусе нет; требуется проверка у специалиста. |
| Отправка с подтверждением | «Отправь в Telegram-чат `<chat_id тестового чата>` текст: “Проверили руководство. Нужны версия продукта и параметры модели”». Замените placeholder числовым ID доступного боту чата, проверьте preview и выберите подтверждение или отмену. |

## Проверки без внешних вызовов

Для локальной проверки установите `uv` и подготовьте проектное окружение:

```powershell
uv sync --extra eval
```

`uv` создаёт `.venv` с Python 3.13 из [pyproject.toml](pyproject.toml).
Полный прогон тестов:

```powershell
.venv/Scripts/python.exe -m pytest tests
```

Короткая проверка основного Telegram/agent/RAG-сценария:

```powershell
.venv/Scripts/python.exe -m pytest tests/chat/test_unified_agent.py tests/test_agent_persistent.py tests/bot tests/unit/test_rag.py -q
```

Эта проверка использует scripted LLM, локальный SQLite, retrieval с
инъецированным retriever и mock Telegram sender. Она покрывает источники,
уточнения, восстановление истории, подтверждение/отмену и ошибки без отправки
реальных сообщений. Наличие настроек без вывода ключей можно проверить так:

```powershell
.venv/Scripts/python.exe scripts/check_env_safety.py
```

## Итоговые материалы

| Материал | Ссылка |
|---|---|
| Указатель актуальной технической документации и статус исторических материалов | [Индекс документации](docs/INDEX.md) |
| Финальная архитектура и семь решений: выбор → альтернативы → причина выбора | [Окончательное техническое описание](docs/final_technical_documentation.md) |
| Финальный persistent LangGraph: узлы, переходы и подтверждения | [Описание графа](docs/final_agent_graph.md), [SVG](docs/final-agent-graph.svg), [Mermaid](docs/final-agent-graph.mmd) |
| Текущий Telegram-сценарий, архитектура интеграции, ограничения и результаты проверок | [Единый агент и retrieval по мануалам](docs/unified_agent.md) |
| Меню, команды, история и подтверждение отправки | [Telegram-бот](docs/telegram_bot.md) |
| RAG: ingestion, retrieval, источники и эксплуатация | [RAG-пайплайн](docs/rag.md) |
| Состав мануалов и пробелы покрытия | [Инвентаризация корпуса](docs/data_inventory.md) |
| Качество финального Telegram-агента | [Оценка 15.10.2026 после исправления собственного веса](docs/telegram_agent_evaluation.md), [ответы и измерения](tests/eval/results/2026-10-15_telegram_agent_self_weight_repeat/summary.json), [34 эталонных вопроса](tests/eval/golden_dataset.json) |
| Промежуточные прогоны | [Пропуск собственного веса](tests/eval/results/2026-10-15_telegram_agent_semantic_fix/acceptance.json), [попытка со сбоями соединения и лимитом итераций](tests/eval/results/2026-10-15_telegram_agent_self_weight_fix/acceptance.json), [оценка предыдущей версии](docs/telegram_agent_evaluation_semantic_fix_2026-10-14.md) |
| Baseline перед смысловыми исправлениями | [Сохранённый отчёт](docs/telegram_agent_evaluation_citation_fix_2026-10-14.md), [неизменённый прогон](tests/eval/results/2026-10-14_telegram_agent_citation_fix_verified/summary.json) |
| Baseline до исправления | [Отчёт 14.10.2026](docs/telegram_agent_evaluation_baseline_2026-10-14.md), [неизменённый исходный прогон](tests/eval/results/2026-10-14_telegram_agent/summary.json) |
| История экспериментов RAG | [RAG evaluation, прогон 12.09.2026](docs/rag_evaluation.md); эти цифры не оценивают финальный агент |
| Сохранение состояния и подтверждение действий | [Persistent agent](docs/agent-persistent-report.md) |
| Обоснование single-agent и результаты сравнения | [Multi-agent report](docs/multi-agent-report.md), [ответы benchmark](docs/multi-agent-benchmark-answers.md) |
| Исторический CLI-дизайн, не схема финального агента | [Архивный архитектурный PDF](docs/architecture-historical-cli.pdf) |

Финальная презентация и демонстрационное видео пока не добавлены в репозиторий;
ссылки на них нужно добавить после подготовки. Исторические benchmark-отчёты
описывают свои датированные прогоны и не заменяют оценку текущей версии.

## Границы текущей версии

Корпус содержит мануалы GTS NX и руководство Civil; 60 HTML-разделов извлечены
из одной CHM-справки. Это 70 исходных файлов, а не 70 независимых руководств.
Установки, лицензирования, тарифов, каталога ошибок и внутреннего FAQ в корпусе
пока нет; его пополнение отложено до следующей версии агента.

При отсутствии сведений или недоступности поиска агент должен явно сообщать
об ограничении; учебная JSON-база не используется в основном Telegram-пути.
Проверка цитат подтверждает существование источника и корректность его номера,
но не достоверность каждого утверждения модели. Проверяйте ответ по мануалу
перед использованием. Подробные границы и фактические проверки описаны в
[отчёте интеграции](docs/unified_agent.md).

## История домашних заданий и учебные режимы

Ниже сохранена история накопительного проекта. Команды в учебных разделах
относятся к отдельным инкрементам; основной запуск готового Telegram-сценария
описан выше.

### CLI: ранний инкремент

Пакет `support_copilot` остаётся учебным CLI, а папки `homework_*` содержат
wrappers и demo-материалы. Его настройки читаются из `.env` через
`support_copilot/config.py`. Offline demo, real mode и Function Calling:

```powershell
uv run python -m support_copilot
uv run python -m support_copilot --real
uv run python -m support_copilot --tools
uv run python -m support_copilot --real --tools
```

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
Файлы сессии по умолчанию содержат только типы событий, счётчики, размеры и
временные HMAC-отпечатки содержимого. Исходные обращения, ответы, пути к
изображениям, аргументы инструментов и тексты ошибок туда не записываются.

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

Переменные `VISION_PROVIDER`, `VISION_MODEL` и `OPENAI_VISION_MODEL` относятся
к этому CLI-сценарию. Для OpenAI порядок выбора:
`OPENAI_VISION_MODEL` → `VISION_MODEL` → `OPENAI_MODEL`.
Telegram-агент сейчас получает подпись к изображению, а не пиксели изображения;
анализ картинки через Vision в эту ветку ещё не подключён.

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

Query pipeline объединяет до 30 dense-кандидатов с точным full-text recall,
оставляет только `RAG_PRODUCT=gts_nx`, ранжирует полный пул до раннего лимита,
удаляет близкие дубли и передаёт LLM максимум пять context fragments. Один
лучший lexical match резервируется, поэтому точное определение не теряется за
семантически похожими соседними разделами. Optional BGE re-ranking и score
guard выполняются до вызова LLM. Ответ содержит цитаты `[1]`, `[2]` и
структурированные sources. Postgres-история участвует в multi-turn retrieval,
чат отправляет источники финальным SSE-событием, Telegram показывает ожидание,
затем обновляет сообщение с ответом с debounce 0.8 секунды, а feedback сохраняет показанные источники.
`POST /documents/upload` возвращает 202 и запускает background UPSERT.

Материалы и команды проверки: `homework_5_5/README.md`,
`homework_5_5/TEACHER_CHECK.md`, `docs/rag.md` и `docs/data_inventory.md`.
