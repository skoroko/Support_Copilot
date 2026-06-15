# Краткая сводка для проверки ДЗ 3.1

В этом инкременте в проект `Support Copilot` добавлен доменный Function Calling
сценарий для техподдержки. Реализован tool `search_knowledge_base(query, product)`,
который ищет релевантные записи в локальной базе знаний и помогает подготовить
ответ инженеру поддержки.

Главная проверяемая логика находится не в одноразовом скрипте, а в основном
пакете `support_copilot`. Папка `homework_3_1` содержит только demo-запуск,
README и результаты прогона для сдачи.

## Как проверить

Offline demo без API-ключа:

```powershell
python homework_3_1\run_tool_call_demo.py
```

Function Calling через основной CLI:

```powershell
python -m support_copilot --tools --scripted-demo --reset-log
```

Real mode, если настроен `.env` с рабочим провайдером:

```powershell
python -m support_copilot --real --tools
```

## Что должно быть видно

В demo есть три кейса из задания:

1. Запрос, где tool точно нужен: вопрос про лицензию. Модель вызывает
   `search_knowledge_base`, handler возвращает запись `KB-LIC-001`, финальный
   ответ основан на базе знаний.
2. Запрос, где tool не нужен: просьба сформулировать короткое сообщение.
   Модель отвечает текстом без вызова tool, код не падает.
3. Пограничный случай: сбой расчета после обновления. Модель вызывает
   `search_knowledge_base`, handler возвращает `KB-CALC-001`, финальный ответ
   перечисляет данные для диагностики.

В JSONL-логе должны быть события:

- `input`;
- `model_request`;
- `tool_call`;
- `tool_result`;
- `tool_skipped`, если tool не нужен;
- `final_answer`;
- `usage`.

## Ключевые файлы

- `homework_3_1/README.md` - описание сдачи и наблюдения по трем кейсам.
- `homework_3_1/run_tool_call_demo.py` - demo-прогон ДЗ 3.1.
- `homework_3_1/results/tool_call_demo.jsonl` - лог demo-прогона.
- `support_copilot/tools/schemas.py` - JSON Schema tool.
- `support_copilot/tools/handlers.py` - Python-handler `search_knowledge_base`.
- `support_copilot/data/knowledge_base.json` - локальная база знаний.
- `support_copilot/infrastructure/tool_calling.py` - полный цикл Function Calling.
- `support_copilot/core/tool_assistant.py` - связка Function Calling с Support Copilot.
- `support_copilot/prompts/system_v1.j2` - system prompt для tool-flow.
- `support_copilot/prompts/tools/search_knowledge_base.md` - description tool.
- `support_copilot/results/tool_call_log.jsonl` - лог основного CLI в режиме `--tools`.
- `tests/test_tool_calling.py` - проверки схемы, handler, config и полного цикла.

## Соответствие критериям самопроверки

| Критерий                                     | Где смотреть                                                     |
| ---------------------------------------------------- | --------------------------------------------------------------------------- |
| Tool описан через JSON Schema             | `support_copilot/tools/schemas.py`                                        |
| Schema валидируется                      | `validate_tool_schema()` и тесты                                    |
| Handler делает реальную работу   | `support_copilot/tools/handlers.py`, чтение `knowledge_base.json` |
| System prompt вынесен в файл             | `support_copilot/prompts/system_v1.j2`                                    |
| Description tool вынесен отдельно     | `support_copilot/prompts/tools/search_knowledge_base.md`                  |
| Полный цикл tool_call реализован | `support_copilot/infrastructure/tool_calling.py`                          |
| Есть assistant turn с `tool_calls`            | `_assistant_message_with_tool_calls()`                                    |
| Есть `role: "tool"` с результатом  | `chat_with_tools()` в `tool_calling.py`                                |
| No-tool сценарий обработан          | `tool_skipped` в JSONL-логе                                          |
| Логируются шаги и usage               | `tool_call_demo.jsonl`, `tool_call_log.jsonl`                           |
| API-ключи не в коде                      | `.env` + `support_copilot/config.py`                                    |

## Проверочные команды

```powershell
python -m compileall support_copilot homework_3_1 tests
python -m unittest
python homework_3_1\run_tool_call_demo.py
python -m support_copilot --tools --scripted-demo --reset-log
```
