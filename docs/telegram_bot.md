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

`bot.__main__` creates `Bot`, `Dispatcher`, registers
handlers, puts `BackendClient` into `dp["backend"]`, and starts polling.

The bot denies all incoming Telegram updates until `BOT_ALLOWED_USER_IDS` is
configured with verified employee Telegram IDs. Private dialogs require the
chat ID to equal the allowed user ID. Groups and supergroups also require the
chat ID in `BOT_ALLOWED_CHAT_IDS`; anonymous senders, channels, bots, and
updates without both a user and chat are denied. IDs may be comma-separated.
`BOT_ADMIN_IDS` grants admin commands only and does not bypass admission: add
an administrator to `BOT_ALLOWED_USER_IDS` as well. The backend service token
authenticates the bot process, while this gate admits individual users.

## Backend Contract

`bot/services/backend_client.py` wraps the 4.1 chat API:

- `get_or_create_chat(owner_external_id, interface)` calls `POST /chats` and
  keeps an in-process chat id cache for repeated calls from the same owner.
  The owner key is `chat.id` for private dialogs and
  `"<chat_id>:<sender_id>"` per admitted member in group chats, so members do
  not share history. The leading segment stays parseable as the Telegram
  delivery chat (`telegram_chat_id_from_owner` in `app/chat/domain.py`),
  which admin notifications and broadcasts use; broadcasts deduplicate
  targets per Telegram chat.
- `send_message(chat_id, content)` calls
  `POST /chats/{chat_id}/messages`, parses SSE `data:` frames, and yields text
  chunks.
- `clear_messages(chat_id)` calls `DELETE /chats/{chat_id}/messages`.

HTTP errors are not hidden in the client. Handlers catch `httpx` exceptions and
show a user-facing message without traceback.

In group chats debounce batches (`bot/handlers/_burst.py`) are keyed by the
owner key, so quick messages from different members never merge into one
request, and answers are sent via `message.reply` to show which question they
belong to. Feedback callbacks derive the owner from the clicking user, so only
the author of a question can rate its answer.

## Commands

- `/start` - creates or reuses the backend chat and shows a short instruction.
- `/help` - lists available commands.
- `/clear` - soft-clears backend history for the current Telegram chat.
