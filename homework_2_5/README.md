# Homework 2.5-2.6 - Support Copilot CLI

Цель: единый интерактивный CLI-помощник техподдержки, который объединяет блоки
2.1-2.4 и дополнительно умеет анализировать изображения из задания 2.6A.

## Что реализовано

- System prompt по РРФО и few-shot примеры для Support Copilot.
- Retry + fallback через `RobustLLMClient` из ДЗ 2.3.
- In-memory кеш через `LLMCache` из ДЗ 2.4.
- Команды `/clear`, `/stats`, `/quit`, `/help`.
- Команда `/image` для анализа пользовательского PNG/JPEG-файла.
- История последних 10 сообщений, включая текстовые результаты анализа изображений.
- Классификация обращений, эскалация, streaming для текстовых ответов и логирование.

## Основной запуск CLI

Рабочая точка входа вынесена в пакет `support_copilot`. Из папки
`HelpDesk_Assistant`:

```powershell
python -m support_copilot
```

Real mode с провайдерами из `.env`:

```powershell
python -m support_copilot --real
```

Real mode со streaming для текстовых ответов и `/image`:

```powershell
python -m support_copilot --real --stream
```

## Совместимый запуск ДЗ 2.5

Файл `homework_2_5/support_cli.py` оставлен как wrapper для сдачи и старых
команд. Из папки проекта `HelpDesk_Assistant`:

```powershell
python .\homework_2_5\support_cli.py
```

Из родительской папки:

```powershell
python .\HelpDesk_Assistant\homework_2_5\support_cli.py
```

## Команды внутри CLI

```text
/help
/stats
/clear
/quit
/image <путь к файлу> | <вопрос по изображению>
```

Пример анализа изображения:

```text
/image ".\homework_2_6\examples\func.png" | Что видно на графике и что запросить у пользователя?
```

Если вопрос не указать, CLI использует стандартный вопрос для обращения в
техподдержку MIDAS.

Если CLI запущен с `--stream`, реальный Vision-ответ по `/image` выводится
потоково. Если провайдер вернул пустой поток до первого токена, CLI пробует
следующего настроенного Vision-провайдера.

## Real Vision API

Для `/image` в real mode используется цепочка Vision-провайдеров:
`OpenAI -> OpenRouter -> Ollama`, если соответствующие ключи и модели заданы в
`.env`.

```dotenv
VISION_PROVIDER=openai
VISION_MODEL=gpt-5.4-mini
OPENAI_VISION_MODEL=gpt-5.4-mini
OPENROUTER_VISION_MODEL=nex-agi/nex-n2-pro:free
OLLAMA_VISION_MODEL=gemma3:4b
OPENROUTER_API_KEY=sk-or-...
```

Можно явно переопределить провайдера при запуске:

```powershell
python -m support_copilot --real --vision-provider openrouter
```

Важно: модель должна поддерживать вход `text+image -> text`. Текстовый
`OPENROUTER_MODEL` используется для обычного чата, а `VISION_MODEL` или
`OPENROUTER_VISION_MODEL` - для обработки изображений.

## Scripted demo

Скриптовое demo осталось только для текстового сценария 2.5:

```powershell
python -m support_copilot --scripted-demo
```

Для задания 2.6 не нужно прогонять подготовленные тестовые картинки: используйте
интерактивный CLI и команду `/image` с реальным файлом пользователя.

## Результаты

Runtime-лог интерактивной сессии по умолчанию пишется в
`support_copilot/results/session_log.txt` и дописывается между запусками. Для
совместимого wrapper-запуска `homework_2_5/support_cli.py` лог пишется в
`homework_2_5/results/session_log.txt`. Чтобы очистить лог перед текущим
запуском, используйте `--reset-log`.
