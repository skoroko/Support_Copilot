# Support Copilot Package

Основной рабочий пакет проекта. Дальнейшие домашние задания лучше подключать
сюда как новые модули, а папки `homework_*` оставлять как учебные wrappers,
demo-скрипты, README и результаты проверок.

## Запуск

Из папки `HelpDesk_Assistant`:

```powershell
python -m support_copilot
```

Real mode:

```powershell
python -m support_copilot --real
```

Real mode со streaming для обычного чата и `/image`:

```powershell
python -m support_copilot --real --stream
```

## Изображения

Внутри CLI:

```text
/image ".\homework_2_6\examples\func.png" | Что видно на графике?
```

Если CLI запущен с `--stream`, real Vision-ответ печатается по токенам.
Fallback между Vision-провайдерами выполняется до первого выведенного токена.

## Структура

- `cli.py` - точка входа CLI и команды.
- `core/vision.py` - анализ изображений, streaming и fallback между Vision-провайдерами.
- `infrastructure/llm.py` - retry/fallback/streaming LLM client.
- `infrastructure/cache.py` - in-memory и Redis cache.

## Логи

По умолчанию все сессии дописываются в `support_copilot/results/session_log.txt`.
Чтобы вручную начать лог заново:

```powershell
python -m support_copilot --reset-log
```
