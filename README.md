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
