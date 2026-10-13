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
prompt-injection and encoded-payload guards. It checks service authentication
before parsing multipart input. It accepts at most four media files and a total raw request body of
`2 * CHAT_MEDIA_MAX_BYTES + 1 MiB`; the limit counts multipart framing and
is enforced on streamed requests without `Content-Length` too. Document
uploads require admin authentication before parsing, one file at most, and a
51 MiB total raw request body limit (50 MiB document plus framing).

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
CHAT_EXPANDED_RESPONSE_TOKENS=1500
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

Every stateful chat request requires the service token and an authenticated
principal. Set the app's `INTERNAL_TOKEN` in your local PowerShell session;
keep its value out of documentation and shell history. Replace the example user
id and interface with the authenticated caller's values.

```powershell
$chatHeaders = @(
  "-H", "X-Internal-Token: $env:INTERNAL_TOKEN",
  "-H", "X-User-ID: support-engineer-1",
  "-H", "X-User-Interface: web"
)
```

Create a chat. The owner and interface are taken from the authenticated headers;
the endpoint does not accept a JSON body:

```powershell
curl.exe @chatHeaders -X POST http://localhost:8000/chats
```

Copy the returned `chat_id` into `$chatId`:

```powershell
$chatId = "<chat-id-from-create-response>"
```

Get chat metadata:

```powershell
curl.exe @chatHeaders "http://localhost:8000/chats/$chatId"
```

Send a text message as multipart form data and receive SSE chunks:

```powershell
curl.exe -N @chatHeaders -X POST "http://localhost:8000/chats/$chatId/messages" `
  -F "content=Please help diagnose this issue"
```

Attach media using the `media` form field. Repeat that field for each file (up
to four files per request); the message text remains the `content` field:

```powershell
curl.exe -N @chatHeaders -X POST "http://localhost:8000/chats/$chatId/messages" `
  -F "content=What does this screenshot show?" `
  -F "media=@C:\path\to\image.png;type=image/png"
```

Ask a follow-up. The backend includes recent history:

```powershell
curl.exe -N @chatHeaders -X POST "http://localhost:8000/chats/$chatId/messages" `
  -F "content=Can you explain the next step?"
```

List messages in chronological order:

```powershell
curl.exe @chatHeaders "http://localhost:8000/chats/$chatId/messages?limit=50"
```

Clear chat history:

```powershell
curl.exe @chatHeaders -X DELETE "http://localhost:8000/chats/$chatId/messages"
```


## Brief answers and confirmed expansion

Telegram answers use a 500 output-token ceiling by default. The graph requests
3–5 key points with complete sentences. Length stops, empty output and defective
mixed-script prose are handled by deterministic compaction, without an automatic
repair generation. Code, technical literals and valid citations remain intact.
A compacted explanation carries a persisted notice distinguishing the length
ceiling from defective text. It offers expansion only when eligible; an expanded
answer instead suggests a separate question about the desired point. It never
describes an output limit as missing information in the user's question.

Assistant `media_refs.answer` stores `mode`, `incomplete`, `expandable`; SSE
`type=message` includes the same `answer_info` with the exact persisted UUID.
`GET /chats/{chat}/messages/{message}/expansion` checks owner, current response,
evidence, pending approval and one-use claim, returning availability and the
configured expanded ceiling without a model call. `POST .../expand` requires
`{"confirmed": true}` (strict boolean; extra fields rejected). It returns the
canonical answer, sources, result message UUID and answer metadata.

Only an eligible latest brief response can expand. The confirmed call uses a
1,500 output-token ceiling and existing evidence, with tools disabled. It follows
the original moderation and append-turn path, recording consent and the detailed
response. A durable graph claim is written before generation, under the shared
chat lock; repeated confirmation returns the saved result or 409 without another
model call. Clear deletes history and checkpoint and invalidates these buttons.
This relies on the documented single API worker; graph and chat persistence are
separate transactions. A failed/unknown generation remains claimed, so a user
must ask a new question rather than silently retrying a potentially billed call.
Token caps are ceilings, not estimates of total input/output cost. This policy
is independent of coding-agent model routing and does not change model IDs/prices.
