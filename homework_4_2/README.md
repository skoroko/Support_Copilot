# ДЗ 4.2 - Telegram-бот как тонкий клиент

## Что реализовано

- `bot/__main__.py` - запуск aiogram 3 polling, `MemoryStorage`, `dp["backend"]`.
- `bot/config.py` - настройки `BOT_TOKEN`, `BACKEND_URL`, `BOT_ADMIN_IDS` через pydantic-settings.
- `bot/services/backend_client.py` - async `httpx` client к `/chats`.
- `bot/handlers/commands.py` - `/start`, `/help`, `/clear`, `/cancel`.
- `bot/handlers/text.py` - обычный текст отправляется в backend, SSE рендерится через `edit_text`.
- `bot/states.py` и `bot/handlers/fsm.py` - FSM-сценарий `/ask`.
- `bot/keyboards/inline.py` - inline-темы из домена Support Copilot.
- `tests/bot/` - тесты BackendClient и FSM-перехода.

## Запуск

```powershell
$env:BOT_TOKEN='123456:telegram-token'
$env:BACKEND_URL='http://127.0.0.1:8000'
python -m bot
```

Backend из ДЗ 4.1 должен быть запущен отдельно:

```powershell
uvicorn app.main:app --reload
```

## Проверка

```powershell
python -m pytest tests/bot -v
python -m pytest
python -m compileall app bot alembic tests homework_4_2
```

`.env` уже находится в `.gitignore`; токен бота не коммитится.
