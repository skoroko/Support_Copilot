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

If commands appear in Telegram but no command receives a reply, check access
configuration before treating command registration as proof of working input:
`.venv/Scripts/python.exe scripts/check_env_safety.py` reports presence of
`BOT_ADMIN_IDS`, `BOT_ALLOWED_USER_IDS` and `BOT_ALLOWED_CHAT_IDS` without
displaying IDs or secrets. An absent user allowlist intentionally rejects all
updates. After changing `.env`, recreate the bot container to reload its
environment; restarting the existing container does not refresh Compose
`env_file` values.

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

The bot shows «Готовлю ответ…» and refreshes `sendChatAction` during preparation.
Before displaying the first nonempty body (including an SSE replacement), it
stops and drains any in-flight typing request, sends a NEW answer message, then
deletes the temporary placeholder best effort. Telegram clears typing on a new
message; edits do not have that documented guarantee. The new message becomes
editable for later deltas and final formatting. Long answers use safe chunks,
with buttons on the last part. An unchanged edit counts as success and never
triggers a duplicate send. A transport without a placeholder buffers until the
final new message. Failures/cancellation cancel and join the refresher; cancelling
during drain does not publish the body. No artificial delay or LLM call is added.

«Подготовить подробный ответ» uses this same renderer while awaiting the single
confirmed expansion request: placeholder and typing, drain, new answer and
placeholder removal. Sources/documents/feedback remain on the final part, with
no recursive expansion button. Backend errors are rendered through the same
lifecycle without source/buttons; cancellation cleans up its waiting message.
Opening the cost notice or cancelling it does not start a waiting status or
generation. The callback is acknowledged before the backend request.

In group chats debounce batches (`bot/handlers/_burst.py`) are keyed by the
owner key, so quick messages from different members never merge into one
request, and answers are sent via `message.reply` to show which question they
belong to. Feedback callbacks derive the owner from the clicking user, so only
the author of a question can rate its answer.

## Navigation and commands

At startup the bot registers Telegram's native command list and commands menu
button. Registration is best effort: a Telegram API failure is logged without
preventing polling, and registration is attempted again if polling restarts.
No admin commands appear in this ordinary user menu.

`/start` and `/menu` show a compact inline home panel with «Примеры вопросов»,
«Как пользоваться», «История», «Ожидающие отправки» and «Новый диалог».
Subpanels provide a return button. Navigation reads do not submit an LLM request
or add messages to backend history. Examples are suggestions to copy into a
message; pressing a navigation button does not submit an example automatically.

«Ожидающие отправки» reads the durable draft and shows its exact recipient and
literal text with the existing confirm/cancel buttons. `/start` also restores
this preview when a draft exists. «Новый диалог» asks for confirmation before
clearing history and pending approval. Cancellation preserves both; confirming
also discards only this owner's unsent debounce batch without generating an
answer. Reset confirmation is single-use, expires after five minutes and must
be reopened after a bot restart. Old confirmation buttons cannot clear a new
dialog; `/menu`, `/start` and `/clear` invalidate an earlier reset confirmation.
In groups each navigation panel belongs to the user who opened it;
another member must open their own `/menu`.

- `/start` - creates or reuses the backend chat and opens the home panel.
- `/menu` - opens the home panel without a backend request.
- `/help` - shows usage instructions with a return button.
- `/clear` - immediately soft-clears this owner's backend history and pending
  approval; it first flushes only this owner's queued message batch.
- `/history` - shows the last ten persisted messages for this owner.

`/clear` also deletes the agent checkpoint under the shared chat turn lock,
invalidating every previously shown approval button. Durable pause/resume
requires SQLite (local default) or PostgreSQL; memory saver is for tests/demo.
See [unified_agent.md](unified_agent.md) for the fully offline integration check.

The default answer ceiling is 500 output tokens (`CHAT_RESPONSE_TOKENS`).
Length stops, empty output and mixed-script defects in prose are compacted
locally into complete text, without an automatic paid repair call. Technical
literals, complete code blocks and grounded citations are preserved.
Parameter labels and headings survive compaction together with complete following
bodies; orphan headings before unfinished tails are removed. Long Telegram
answers prefer paragraph, line and sentence boundaries, keeping headings with
their bodies and paired Markdown together when they fit. Chunking preserves
every original character; indivisible overlong spans use a hard fallback.
Buttons remain on the last chunk, and approval previews retain literal text.

Eligible brief answers show **Подробнее** alongside documents and feedback.
It opens a cost notice: a detailed answer is an additional AI request that
consumes tokens and may increase processing cost. **Подготовить подробный ответ**
confirms a single generation with a 1,500 output-token ceiling
(`CHAT_EXPANDED_RESPONSE_TOKENS`), based on existing evidence with tools disabled.
**Оставить краткий** changes no history and makes no model call. These ceilings
are not monetary estimates and do not change runtime models or provider prices.

Callbacks carry exact compact chat/message UUIDs (`exp:a`, `exp:y`, `exp:n`),
fit Telegram's 64-byte limit and use the clicking owner's explicit principal,
including group member isolation. Callback acknowledgement precedes backend
work. Feedback preserves documents and the expansion affordance. Expanded
answers have sources and feedback, with no further expansion loop. Only panel
controls are removed on confirmation/cancel; panel-edit failure does not prevent
delivery of an already saved detailed answer. Owner/stale/clear checks fail safe.
POST is never retried automatically. Durable claims prevent another generation
on double click, timeout or process restart; missing saved results return 409.

Offline menu runtime coverage uses real aiogram `Dispatcher`, `Update` and
callback/message routing with a mocked Telegram transport and backend:
`.venv/Scripts/python.exe -m pytest tests/bot/test_menu.py -q`. No live Telegram visual check or
outgoing user-message delivery is performed by these tests.

Verified 2026-10-13:

- `.venv/Scripts/python.exe -X utf8 -m pytest tests/bot tests/chat/test_unified_agent.py -q`
  — PASS: **150 passed**.
- Production `bot.handlers.router` runtime smoke with admission middleware,
  synthetic Telegram updates and mocked transport — PASS: home panel,
  group owner isolation, denied user, and ordinary-question debounce routing.
- Ruff on all changed Python files and `git diff --check` — PASS.

The commands menu uses Telegram's supported
[`setMyCommands`](https://core.telegram.org/bots/api#setmycommands) and
[`setChatMenuButton`](https://core.telegram.org/bots/api#setchatmenubutton).
It is registered when the updated bot process starts; existing running
processes must be restarted or redeployed to load the new handlers.
