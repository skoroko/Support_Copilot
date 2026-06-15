# Домашнее задание 3.1: Function Calling

Реализован доменный tool для Support Copilot:

- `search_knowledge_base(query, product)` ищет по локальной JSON-базе знаний `support_copilot/data/knowledge_base.json`;
- JSON Schema находится в `support_copilot/tools/schemas.py`;
- обработчик находится в `support_copilot/tools/handlers.py` и валидирует аргументы через Pydantic;
- API-ключи, модели, base URL, timeout/retry настройки читаются через `support_copilot/config.py` на базе `pydantic-settings`;
- system prompt вынесен в `support_copilot/prompts/system_v1.j2`;
- description tool вынесен в `support_copilot/prompts/tools/search_knowledge_base.md`;
- полный цикл tool call реализован в `support_copilot/infrastructure/tool_calling.py`;
- JSONL-лог пишется в `homework_3_1/results/tool_call_demo.jsonl`.

## Запуск

Offline demo без API-ключа:

```powershell
python homework_3_1/run_tool_call_demo.py
```

Тот же Function Calling flow подключен к основному CLI:

```powershell
python -m support_copilot --tools
```

Real mode через OpenAI-compatible provider chain из `.env`:

```powershell
python homework_3_1/run_tool_call_demo.py --real
```

## Проверенные кейсы

### 1. Запрос точно требует tool

Запрос: `Найди в базе знаний, что запросить у клиента, если лицензия не активируется второй день.`

Модель вызвала `search_knowledge_base` с аргументами `product=licensing` и исходным текстом запроса в `query`. Handler прочитал JSON-базу знаний и вернул запись `KB-LIC-001`. Финальный ответ перечислил данные для диагностики лицензии и условие эскалации.

### 2. Запрос точно не требует tool

Запрос: `Сформулируй короткое вежливое сообщение, что инженер принял обращение в работу.`

Модель не вызвала tool, потому что вопрос просит только переформулировать текст и не требует фактов из базы знаний. Код обработал прямой текстовый ответ без падения. В логе есть событие `tool_skipped`.

### 3. Пограничный случай

Запрос: `Клиент пишет: после обновления расчет падает. Что ответить сначала?`

Модель решила вызвать `search_knowledge_base`, потому что в вопросе есть технический симптом и потенциально известный порядок диагностики. Handler вернул `KB-CALC-001`. Финальный ответ предложил запросить версии, файл модели или минимальный пример, лог расчета, текст ошибки и шаги воспроизведения.

## Логирование

Каждый запуск пишет JSONL-события:

- `input`;
- `model_request`;
- `tool_call`;
- `tool_result`;
- `tool_skipped`, если tool не нужен;
- `final_answer`;
- `usage` с `total_tokens`.

Дополнительная сводка последнего demo-запуска пишется в `homework_3_1/results/demo_summary.md`.
