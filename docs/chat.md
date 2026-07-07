# Stateful Chat Backend

Homework 4.1 adds server-side chat history for Support Copilot. The stable API
contract is `/chats/{chat_id}/messages`: clients send only a chat id and a new
message, while the backend loads history, builds context, streams the LLM answer,
and persists both user and assistant messages.

## Architecture

```mermaid
flowchart LR
    Client[Web / CLI / Telegram client]
    Routes[app/chat/routes.py]
    Service[ChatService]
    Repo[ChatRepository]
    Json[(JSONL files)]
    Pg[(Postgres)]
    LLM[AsyncOpenAI-compatible client]

    Client --> Routes
    Routes --> Service
    Service --> Repo
    Repo --> Json
    Repo --> Pg
    Service --> LLM
```

`routes.py` contains only HTTP contracts and SSE formatting. `ChatService`
owns domain flow: append the user message, load history, build context, apply a
token budget, stream the LLM response, and append the final assistant message.
`ChatRepository` is a `typing.Protocol`, so tests and future clients can provide
fake repositories without inheritance.

`POST /chats/{chat_id}/messages` reuses the project input security validator
before calling the LLM, so the new stateful endpoint does not bypass the M3
prompt-injection and encoded-payload guards.

## Context Strategy

The project uses `CHAT_CONTEXT_STRATEGY=sliding` for the first stable backend
increment. Support Copilot is currently an internal support assistant, and the
near-term workflows are FAQ, issue triage, and escalation drafts where the last
few turns are usually enough. Sliding window is predictable, cheap, easy to
test, and keeps the M4 API stable before RAG and long-dialog summarization are
added in later modules.

Configuration:

```env
CHAT_CONTEXT_STRATEGY=sliding
CHAT_CONTEXT_WINDOW=10
CHAT_MODEL_CONTEXT_TOKENS=128000
CHAT_RESPONSE_TOKENS=500
CHAT_SAFETY_MARGIN_TOKENS=500
```

Token counting uses `tiktoken.get_encoding("o200k_base")`, with ChatML overhead
`+4` per message and `+2` total. `fit_to_budget()` trims oldest non-system
messages first and preserves the system prompt.

## Storage

Default local storage:

```env
CHAT_REPOSITORY=json
CHAT_STORAGE_DIR=./var/chats
```

JSON layout:

```text
var/chats/chats/<chat_id>/chat.json
var/chats/chats/<chat_id>/messages.jsonl
```

Postgres storage:

```env
CHAT_REPOSITORY=postgres
CHAT_DATABASE_URL=postgresql+asyncpg://support:support@localhost:5432/support_copilot
```

Run migrations before starting in Postgres mode:

```powershell
alembic upgrade head
uvicorn app.main:app --reload
```

`DELETE /chats/{chat_id}/messages` is a soft delete. JSONL appends a
`{"type":"soft_delete"}` marker; Postgres sets `deleted_at` on existing message
rows.

## Endpoints

Create a chat:

```powershell
curl -X POST http://localhost:8000/chats `
  -H "Content-Type: application/json" `
  -d '{"owner_external_id":"test-1","interface":"cli","system_prompt":"Отвечай как помощник техподдержки MIDAS."}'
```

Get chat metadata:

```powershell
curl http://localhost:8000/chats/<chat_id>
```

Send a message and receive SSE chunks:

```powershell
curl -N -X POST http://localhost:8000/chats/<chat_id>/messages `
  -H "Content-Type: application/json" `
  -d '{"content":"Привет, меня зовут Аня"}'
```

Ask a follow-up. The backend will include recent history:

```powershell
curl -N -X POST http://localhost:8000/chats/<chat_id>/messages `
  -H "Content-Type: application/json" `
  -d '{"content":"Как меня зовут?"}'
```

List messages in chronological order:

```powershell
curl "http://localhost:8000/chats/<chat_id>/messages?limit=50"
```

Clear chat history:

```powershell
curl -X DELETE http://localhost:8000/chats/<chat_id>/messages
```
