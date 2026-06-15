# Support Copilot Package

Основной рабочий пакет проекта. В него подключаются накопительные инкременты
домашних заданий, а папки `homework_*` остаются учебными wrappers, demo-скриптами,
README и результатами проверок.

## Запуск

Из папки `HelpDesk_Assistant`:

```powershell
python -m support_copilot
```

Real mode через provider chain из `.env`:

```powershell
python -m support_copilot --real
```

Streaming для обычного чата и `/image`:

```powershell
python -m support_copilot --real --stream
```

Function Calling mode с доменным tool `search_knowledge_base`:

```powershell
python -m support_copilot --tools
```

Real Function Calling mode:

```powershell
python -m support_copilot --real --tools
```

Scripted demo для Function Calling:

```powershell
python -m support_copilot --tools --scripted-demo --reset-log
```

## Настройки

API-ключи, модели, base URL, timeout/retry настройки, Vision-настройки и пути логов
читаются через `support_copilot/config.py`.

Реальные секреты хранятся только в `.env`, например:

```env
OPENAI_API_KEY=
OPENAI_MODEL=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=
OLLAMA_MODEL=qwen3:1.7b
```

`.env.example` содержит только шаблон переменных. Ключи нельзя записывать в код,
README, тесты или логи.

## Function Calling

Режим `--tools` использует полный OpenAI-compatible цикл tool calling:

1. Отправляет `messages + tools`.
2. Парсит `tool_calls`.
3. Выполняет Python-handler.
4. Добавляет assistant-сообщение с `tool_calls` и `role: "tool"` с результатом.
5. Отправляет второй запрос за финальным ответом.

Основные файлы:

- `core/tool_assistant.py` - обертка Support Copilot для Function Calling.
- `infrastructure/tool_calling.py` - полный цикл tool call, retry/fallback provider logic.
- `tools/schemas.py` - JSON Schema tool `search_knowledge_base`.
- `tools/handlers.py` - обработчик `search_knowledge_base`.
- `data/knowledge_base.json` - локальная учебная база знаний.
- `prompts/system_v1.j2` - system prompt для tool-flow.
- `prompts/tools/search_knowledge_base.md` - description tool.

## Изображения

Внутри CLI:

```text
/image ".\homework_2_6\examples\func.png" | Что видно на графике?
```

Если CLI запущен с `--stream`, real Vision-ответ печатается по токенам. Fallback
между Vision-провайдерами выполняется до первого выведенного токена.

## Структура

- `cli.py` - точка входа CLI, команды и режим `--tools`.
- `config.py` - централизованные настройки через `pydantic-settings` с fallback.
- `core/vision.py` - анализ изображений, streaming и fallback.
- `core/tool_assistant.py` - доменный assistant для Function Calling.
- `infrastructure/llm.py` - retry/fallback/streaming LLM client.
- `infrastructure/tool_calling.py` - OpenAI-compatible Function Calling client.
- `infrastructure/cache.py` - in-memory и Redis cache.
- `tools/` - схемы и обработчики tools.
- `prompts/` - версионируемые prompt-файлы.
- `data/` - локальные данные для tools.

## Логи

Обычный CLI пишет лог в:

```text
support_copilot/results/session_log.txt
```

Function Calling mode пишет JSONL-лог в:

```text
support_copilot/results/tool_call_log.jsonl
```

Начать лог заново:

```powershell
python -m support_copilot --reset-log
python -m support_copilot --tools --reset-log
```
