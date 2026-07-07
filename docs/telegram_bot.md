# Telegram Bot Thin Client

Homework 4.2 adds a Telegram bot on aiogram 3. The bot is intentionally thin:
it does not call LLM providers and does not store dialog history. All stateful
chat behavior remains in the FastAPI backend from homework 4.1.

## Runtime

```powershell
$env:BOT_TOKEN='123456:telegram-token'
$env:BACKEND_URL='http://127.0.0.1:8000'
python -m bot
```

`bot.__main__` creates `Bot`, `Dispatcher(storage=MemoryStorage())`, registers
handlers, puts `BackendClient` into `dp["backend"]`, and starts polling.

## Backend Contract

`bot/services/backend_client.py` wraps the 4.1 chat API:

- `get_or_create_chat(owner_external_id, interface)` calls `POST /chats` and
  keeps an in-process chat id cache for repeated calls from the same Telegram
  chat.
- `send_message(chat_id, content)` calls
  `POST /chats/{chat_id}/messages`, parses SSE `data:` frames, and yields text
  chunks.
- `clear_messages(chat_id)` calls `DELETE /chats/{chat_id}/messages`.

HTTP errors are not hidden in the client. Handlers catch `httpx` exceptions and
show a user-facing message without traceback.

## Commands

- `/start` - creates or reuses the backend chat and shows a short instruction.
- `/help` - lists available commands.
- `/clear` - soft-clears backend history for the current Telegram chat.
- `/cancel` - clears the active FSM state.
- `/ask` - starts a guided support question flow with domain topics.

The `/ask` topics are tied to Support Copilot: billing, installation, API
errors, licensing, and escalation.
