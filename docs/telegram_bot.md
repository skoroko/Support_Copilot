# Telegram Bot Thin Client

Homework 4.2 adds a Telegram bot on aiogram 3. The bot is intentionally thin:
it does not call LLM providers and does not store dialog history. All stateful
chat behavior remains in the FastAPI backend. All ordinary Telegram messages
now use the persistent agent through the same chat facade; there is no `/agent`
mode switch. The agent searches the production manual corpus, drafts replies,
and pauses before any requested external delivery.

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
  keeps an in-process cache. The server resolves the same durable Telegram
  owner/interface chat after a bot restart (JSON and PostgreSQL repositories).
  Existing chats are reused; new chats use a deterministic UUID under a
  repository lock / primary-key conflict guard.
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
- `pending_approval(chat_id)` calls `GET /chats/{chat_id}/approval`; `/start`
  restores the exact pending preview and buttons after a restart.
- `resume_approval(...)` posts the exact opaque draft ID and boolean decision
  to `POST /chats/{chat_id}/approvals/{approval_id}`. The server verifies chat
  ownership and the current draft under the same per-chat turn lock, then
  resumes the durable interrupt. Other owners receive 404; stale/repeated IDs
  receive 409. The caller cannot change recipient or text during approval.

An SSE `approval` event carries only `approval_id`, recipient `chat_id`, and
the exact outgoing `text`. The bot shows that preview with «Подтвердить» and
«Отменить». The callback encodes chat UUID + draft UUID in fewer than 64 bytes;
it derives the authenticated owner from the clicking user, including groups.
Long replies/previews split before HTML rendering, with buttons on the final
part and no text truncation. Delivery/cancellation/unknown status is reported
from the observed tool result, without another model call.

Media still uses the multipart extraction and input moderation path. Captions,
voice transcripts and extracted document text reach the agent; raw image data
is excluded from checkpoints. Images without a description get the existing
explicit request for a text description. Output and the exact outgoing draft
are moderated before display/approval. Sources and feedback remain persisted
on assistant messages. RAG unavailability produces an explicit refusal to
invent documentation, rather than a synthetic fallback.

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
- `/history` - shows the last ten persisted messages for this owner.

`/clear` also deletes the agent checkpoint under the shared chat turn lock,
invalidating every previously shown approval button. Durable pause/resume
requires SQLite (local default) or PostgreSQL; memory saver is for tests/demo.
See [unified_agent.md](unified_agent.md) for the fully offline integration check.
