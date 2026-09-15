# Homework 6.4 - Persistent LangGraph agent

## 1. Backend selection

`app/services/agent_persistent.py` keeps the Б6.3 graph untouched and adds a
new `build_agent(checkpointer)` factory. `AGENT_CHECKPOINTER=sqlite` is the
default for local development and stores state in `./var/agent_checkpoints.sqlite`
(overridable with `AGENT_SQLITE_PATH`). `memory` is available for short-lived
tests; `postgres` is the compose default and uses the existing Support Copilot
Postgres database. A stable client `thread_id` is required by the HTTP contract
and is never generated per request. The API hashes it together with the
authenticated `X-User-ID`, so resume reaches the same lineage without exposing
another caller's checkpoint namespace.

For both persistent savers `await saver.setup()` is executed once inside the
FastAPI lifespan. It is deliberately absent from the endpoint path.

## 2. Postgres in Docker Compose

The `app` service now sets `AGENT_CHECKPOINTER=postgres`, a psycopg-v3
`AGENT_POSTGRES_URI` assembled from the same `POSTGRES_*` values as the
existing `postgres` service, and `LANGGRAPH_STRICT_MSGPACK=true`. No second
database or service is created. An explicit `AGENT_POSTGRES_URI` can override
the assembled URI when credentials require URL encoding.

Alembic's `include_name` excludes exactly `checkpoints`, `checkpoint_writes`,
`checkpoint_blobs`, and `checkpoint_migrations`; `AsyncPostgresSaver.setup()`
owns those tables while Alembic keeps owning domain tables.

The first Docker smoke run exposed a real packaging bug: `app.main` now imports
the persistent agent, whose shared knowledge-base tool imports the
`support_copilot` package, but the Dockerfile did not copy that package. The
image failed before FastAPI startup with `ModuleNotFoundError`. Adding
`COPY support_copilot ./support_copilot` plus a narrow `.dockerignore`
allowlist for config/tools/prompts/data fixed the production import boundary
without copying local result logs. The corrected service became healthy and
the real shared Postgres database returned:

```text
table_name
-----------------------
checkpoint_blobs
checkpoint_migrations
checkpoint_writes
checkpoints
(4 rows)
```

Commands actually used were `docker compose up -d --build app`, `docker
compose ps app`, and a read-only `information_schema.tables` query through
`docker compose exec -T postgres psql`.

## 3. Dangerous action and idempotency boundary

The dangerous domain action is `send_telegram_message`: a delivery can notify
the wrong support chat and is externally observable. The model may request the
tool, but never calls its handler directly. The graph has two explicit nodes:
`prepare_telegram_message -> confirm_and_execute_telegram_message`.

**Before `interrupt`:** argument validation and deterministic draft creation
(`chat_id`, `text`, `tool_call_id`, `idempotency_key`, `sent=False`) only. This
`prepare` node is idempotent and has no API/network side effect. **After
`interrupt`:** only an approved resume may invoke the injected `send_message`
adapter and set `sent=True`. The production adapter passes the deterministic
key to the authenticated bot `POST /notify` endpoint. The bot claims that key
with Redis `SET NX` before Telegram delivery; Redis uses AOF with synchronous
fsync in compose. A repeated resume therefore returns a deduplicated success
instead of sending twice. If Telegram fails before delivery completes, the
claim is retained as `unknown`; an automatic retry is blocked until an
operator reconciles the outcome. Unit tests inject doubles at both network
boundaries, so no Telegram call occurs offline.

Every tool call in one assistant turn receives a matching `ToolMessage`.
Mixed safe/dangerous batches execute or reject the safe calls before approval,
while the selected dangerous call is paired only after approve/reject. At the
six-turn ceiling, all still-pending calls receive explicit rejection results
before the escalation message, keeping the persisted provider history valid.

The ReAct safety ceiling from Б6.3 is preserved: after six model turns the
graph emits an explicit escalation answer instead of continuing indefinitely.

The local Python 3.10 runtime exposed a real integration issue: LangGraph
1.2 dispatches a synchronous node through an executor without propagating the
interrupt contextvar. The module re-installs the supplied node configuration
only for that compatibility failure, while still using the public
`interrupt()` and `Command(resume=...)` APIs. Raising the project runtime to
Python 3.11+ removes this compatibility path.

## 4. Offline interrupt and approval log

Actual output from `PYTHONUTF8=1 .venv-eval\Scripts\python.exe -B
scripts\time_travel_demo.py` (checkpoint ids are generated per run):

```text
INTERRUPT [{"value": {"type": "approve_send_telegram_message", "preview": {"chat_id": "4242", "text": "Инцидент передан инженеру.", "tool_call_id": "demo-telegram-call", "idempotency_key": "<sha256>"}, "user_role": "write-with-approve"}}]
HISTORY checkpoint_id next
1f1b0c8f-95b4-656a-8002-f27df6bebcc7 confirm_and_execute_telegram_message
1f1b0c8f-95b4-6569-8001-c9d6043abedb prepare_telegram_message
1f1b0c8f-95af-66fd-8000-42603b088e70 call_model
1f1b0c8f-95ab-6c25-bfff-99c4890def12 __start__
PAST {"checkpoint_id": "1f1b0c8f-95b4-656a-8002-f27df6bebcc7", "draft": {"chat_id": "4242", "text": "Инцидент передан инженеру.", "tool_call_id": "demo-telegram-call", "idempotency_key": "<sha256>"}, "sent": false, "next": ["confirm_and_execute_telegram_message"]}
REJECT {"sent": false, "delivery_count": 0}
APPROVE {"sent": true, "delivery_count": 1}
```

The first state visibly pauses at the approval node. A `False` resume cannot
call the adapter; a `True` resume calls it once.

## 5. Time travel and branching

`scripts/time_travel_demo.py` uses `AsyncSqliteSaver.from_conn_string(":memory:")`
and no provider/network client. It prints 3-5 `aget_state_history` rows, then
reads the old checkpoint by `thread_id + checkpoint_id`. That snapshot contains
the prepared draft, `sent=false`, and `next=confirm_and_execute_telegram_message`.

The rejection and approval alternatives intentionally use different stable
thread ids (`demo-reject` and `demo-approve`) with identical input. A second
`Command(resume=...)` for one interrupt lineage is not a different decision:
the first resume is checkpointed, so replaying it with another value would be
misleading.

## 6. Streaming

`POST /agent/stream` accepts exactly one of bounded user-only `input` or
`resume`, plus stable `thread_id` and a `user_role`. It requires
`X-Internal-Token` and `X-User-ID`; `full` additionally requires
`X-Admin-Token`. It invokes `graph.astream(..., stream_mode=["updates",
"messages"])` and serializes each value as `data: {json}\n\n` with
`text/event-stream`. `updates` makes node progress and `__interrupt__` visible;
`messages` leaves room for model token events. `astream_events(v2)` was not
used because its event tree is broader and more expensive for this minimum SSE
endpoint. Provider/graph failures finish with a safe `agent_stream_failed` SSE
event without exception text.

The offline endpoint test emitted the following SSE shape using a fake graph:

```text
data: {"type":"updates","payload":{"call_model":{"messages":["model started"]}}}

data: {"type":"messages","payload":[{"content":"Подтвердите отправку"},{"node":"call_model"}]}

data: {"type":"updates","payload":{"__interrupt__":[{"value":{"type":"approve_send_telegram_message"}}]},"interrupt":[{"value":{"type":"approve_send_telegram_message"}}]}

data: {"type":"done"}
```

The authenticated live smoke test used a unique thread, `read-only`, and an
instruction not to call tools. OpenAI returned the Russian sentence
`Инженер технической поддержки помогает пользователям решать технические
проблемы, настраивает и диагностирует сервисы и при необходимости эскалирует
сложные случаи.` through `messages -> updates -> done`. HTTP status was 200;
there were no tool calls and no Telegram delivery.

For a running local API, equivalent calls must include the authenticated
service headers (values intentionally omitted here):

```powershell
curl.exe -N -X POST http://localhost:8000/agent/stream -H "Content-Type: application/json" -H "X-Internal-Token: <configured>" -H "X-User-ID: alice" -d '{"thread_id":"midas-alice-license","input":{"messages":[{"role":"user","content":"Передай инженеру проверку лицензии"}]}}'
curl.exe -N -X POST http://localhost:8000/agent/stream -H "Content-Type: application/json" -H "X-Internal-Token: <configured>" -H "X-User-ID: alice" -d '{"thread_id":"midas-alice-license","resume":true}'
```

## 7. Permission policy

`read-only` cannot send, `write-with-approve` always pauses for an explicit
approval, and `full` is available only when both internal and administrative
credentials are valid. The endpoint defaults to `write-with-approve`.
Authenticated caller identity is part of the private checkpoint namespace, so
one caller cannot resume another caller's pending action by reusing its public
thread id. System/tool/assistant roles are rejected at the HTTP boundary.

## 8. Remaining fragile points

The delivery path now has at-most-once Redis-backed deduplication, but it still
needs a durable business delivery receipt/outbox if guaranteed delivery is
required: a crash after claiming a key but before Telegram accepts the message
prefers a missed delivery over a duplicate. Production should also define
checkpoint retention/encryption, per-identity rate and concurrency limits,
redacted audit logging, and a repeatable Postgres integration test. The Python
3.10 interrupt compatibility shim should be removed after a verified Python
3.11+ upgrade.
