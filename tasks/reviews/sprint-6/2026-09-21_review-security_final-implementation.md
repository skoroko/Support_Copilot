# Security review: final implementation

Date: 2026-09-21
Baseline: `a519aa7b682b1872824f71e4b9e83db650065ed8`
Scope: externally reachable FastAPI and Telegram entry points, authentication
and authorization, chat ownership, provider-cost boundaries, uploads/parsers,
streaming filters, tracing exposure, secrets/tokens, persistence and agent tool
boundaries. Dependency advisories and general code quality are covered by
separate reports.

## Result

**FAIL** — five confirmed P1 vulnerabilities and two P2 vulnerabilities remain
open.

| Severity | Count |
| --- | ---: |
| P0 | 0 |
| P1 | 5 |
| P2 | 2 |
| P3 | 0 |

## Actual trust boundaries and data flow

1. **LAN/browser/CLI → app:8000.** `compose.yaml:14-15` publishes the app on
   host interfaces. `/chat`, `/chat/stream`, `/rag/query`,
   `/documents/upload` and `/chats/**` are unauthenticated. Only
   `/chats/admin/**` uses `X-Admin-Token`; `/agent/stream` uses
   `X-Internal-Token`, with an additional admin gate for the `full` role.
2. **Telegram user → Telegram API → bot → app.** Normal bot requests to chat
   routes have no service-auth header. The app cannot distinguish the bot from
   a direct caller and trusts caller-supplied `owner_external_id`.
3. **App → bot:9000/notify → Telegram.** This hop uses a shared bearer token,
   while `compose.yaml:104-105` also publishes the bot API on host interfaces.
4. **App → OpenAI/embedding providers.** Public chat, RAG and media routes can
   invoke billable external operations and can transfer uploaded content.
5. **App → Postgres/JSON/Redis/Qdrant/filesystem.** These stores hold chat
   metadata, messages, moderation/feedback, media references, checkpoints and
   the RAG corpus. Images can be persisted as base64 data URLs.
6. **App → Phoenix.** OpenAI/LlamaIndex instrumentation is enabled by default;
   RAG questions are stored as raw trace input, and the Phoenix UI/storage is
   published on host port 6006.
7. **Standalone `support_copilot/api.py`.** This is another unauthenticated
   streaming entry point. `.dockerignore:33-41` keeps it out of the current
   Docker image, but it remains locally runnable.

## Confirmed vulnerabilities

### S-01 — P1 — Broken chat ownership enables cross-user access and Telegram notification abuse

**Evidence.** `app/chat/routes.py:38-45` lets an unauthenticated caller create a
chat with an arbitrary `owner_external_id`. Lines 48-60, 140-178 and 205-210
read messages/sources, change feedback and clear history using only a UUID,
without a principal/owner check. Lines 181-202 append a system message and, when
`notify=true`, convert the untrusted owner identifier to a Telegram chat ID and
call the privileged notifier. `tests/chat/test_routes.py:75-108` codifies
unauthenticated CRUD as the current contract.

**Reachable attack.** A caller creates a chat with a known Telegram chat ID,
uses the returned UUID, then posts an arbitrary system message with
`notify=true`. `app/services/notifier.py:24-28` supplies the app's internal
token to `bot/web.py:25-48`, bypassing direct protection on `/notify`. A leaked
or guessed chat UUID also exposes another user's history, feedback and delete
operation.

**Impact.** Cross-user disclosure/modification/deletion and arbitrary messages
sent through the trusted Telegram bot.

**Smallest reasonable correction.** Require an authenticated principal for the
entire `/chats` router, derive external ownership from verified identity, bind
all repository operations to owner/tenant, and move system-message notification
to a separately authenticated internal/admin route.

### S-02 — P1 — Public billable and mutating APIs have no authentication or quotas

**Evidence.** `/chat` and `/chat/stream` are defined at
`app/routers/chat.py:27-72`; `/rag/query` at `app/routers/rag.py:13-44`;
`/documents/upload` at `app/routers/documents.py:85-122`; and the standalone
provider stream at `support_copilot/api.py:44-77`. The caller can select a model
and provide an unbounded message list, with output up to 16k tokens under
`app/schemas/chat.py:55-58`. No user/service authentication, rate limit,
concurrency budget, token budget or storage quota protects these routes.

**Impact.** A remote caller can consume provider budget and workers, create
unbounded state, and trigger expensive ingestion. When upload is enabled, the
same route can persist malicious documents into RAG, creating a poisoning and
indirect prompt-injection path.

**Smallest reasonable correction.** Add mandatory service/user authentication,
per-principal rate/concurrency/token/storage budgets, a server-side model
allowlist and message-count bounds. Restrict ingestion to an admin/ingestion
role and a bounded worker queue.

### S-03 — P1 — Unbounded chat media upload permits remote memory/CPU denial of service

**Evidence.** `app/chat/routes.py:63-70` accepts multipart media without a size
bound. `app/chat/media.py:37-38` materializes the entire upload with
`await media.read()`; lines 42-47 base64-encode a complete image; lines 65-71
and 94-135 parse PDF/DOCX in process. MIME type is caller-controlled. Because
the caller can create a chat through S-01, knowledge of an existing UUID is not
a prerequisite.

**Impact.** One or more large uploads can exhaust RAM/CPU, expand persistent
JSON/DB state and expose in-process parser surfaces.

**Smallest reasonable correction.** Enforce reverse-proxy and application byte
limits before parsing/provider calls, stream reads under a hard cap, validate
magic bytes, dimensions/pages and expanded archive size, isolate/time-bound
parsers, and avoid persisting raw base64 in chat history.

### S-04 — P1 — Raw support questions are stored in a publicly published Phoenix service

**Evidence.** `app/observability/tracing.py:15-21,39-56` enables tracing and
OpenAI/LlamaIndex instrumentation by default. Lines 72-90 store the full RAG
question in the `input.value` span attribute. `compose.yaml:29-31` enables the
collector for the app, while `compose.yaml:132-140` publishes Phoenix UI and
storage without repository-defined authentication or loopback-only binding.

**Impact.** Any party able to reach host port 6006 can potentially inspect
support questions, trace metadata and captured provider data. Persistence
extends the disclosure window beyond the original request.

**Smallest reasonable correction.** Remove the host port or bind it to
loopback. If shared access is required, place Phoenix behind authenticated TLS.
Redact or hash `input.value` and explicitly disable prompt/output capture where
supported.

### S-05 — P1 — Known default/placeholder bearer tokens fail open

**Evidence.** `app/core/config.py:81-87` defines known token defaults, and
`app/admin/auth.py:15-16` accepts the configured default as valid. Bot config
also supplies a known admin default at `bot/config.py:27-31`.
`.env.example:104,107` documents public placeholder values. The agent rejects
only the literal `replace-with-internal-token` at
`app/routers/agent.py:98-100`; the different placeholder documented in the
example passes that check. `bot/web.py:22-28` compares the notify token but does
not reject weak/placeholder configuration. Safe local verification showed that
app and bot objects can be created with known defaults; no actual `.env` value
was read.

**Impact.** A missed environment variable or blindly copied example can expose
admin user listing/broadcast, bot notifications and the internal agent boundary
to anyone who knows the repository defaults.

**Smallest reasonable correction.** Make both tokens mandatory and reject all
documented placeholders, short values and equality between privileged tokens
at startup. Extend the status-only environment safety check to report their
configuration without printing values.

### S-06 — P2 — Streaming output filtering is bypassed by chunk boundaries

**Evidence.** `app/services/llm.py:105-123` filters each provider delta
independently and supplies an empty `system_prompt` on the streaming path. The
canary and PII detectors in `app/services/security/output_filter.py:27-45`
require a complete pattern. An offline probe confirmed that `CANARY_` plus
`deadbeef`, and `alice@` plus `example.com`, pass when split across deltas.

**Impact.** Canary/system-prompt content and PII can be emitted piecewise even
though the complete string would be blocked or redacted.

**Smallest reasonable correction.** Maintain a rolling cross-chunk buffer; for
strict guarantees, buffer and validate the complete response before emitting
it. PII redaction state must span chunk boundaries.

### S-07 — P2 — Public chat accepts attacker-controlled system, assistant and tool roles

**Evidence.** `app/schemas/chat.py:19-21` accepts `system`, `assistant` and
`tool` messages. `app/routers/chat.py:80-95` validates only `user` messages,
then lines 104-110 prepend the trusted security prompt without removing the
caller's privileged-role messages.

**Impact.** A direct caller can inject additional system/tool context and
weaken the prompt-injection boundary. In combination with S-06, this increases
the likelihood of policy/canary leakage.

**Smallest reasonable correction.** Accept only user messages on the public
contract. Reconstruct assistant/tool/system history from trusted server state
and use strict schemas that forbid extra roles/fields.

## Questions and concrete risks

- `APP_CORS_ORIGINS=*` is the default at `app/core/config.py:51-54` and the app
  allows all methods/headers at `app/main.py:151-158`, making unauthenticated
  endpoints easier to drive from a browser.
- Repository evidence does not establish a TLS reverse proxy, firewall policy
  or production Phoenix authentication.
- `X-User-ID` is logged without a verified identity binding or documented
  length/redaction limit (`app/main.py:165-172`).
- No retention, encryption-at-rest or backup-access policy was found for chat,
  media, checkpoints or Postgres volumes.
- Broadcast claim is non-atomic (`app/chat/repositories/pg_repo.py:226-243`,
  `bot/background.py:19-30`); multiple workers can duplicate a broadcast.
- Retrieved RAG documents are not explicitly framed as untrusted
  instructions. Public upload would make corpus poisoning persistent.
- OpenInference prompt/output capture behavior for the installed version was
  not fully verified; raw RAG `input.value` is confirmed directly in code.

## Good controls observed

- Admin and bot token comparisons use constant-time comparison.
- `/agent/stream` has bounded schemas, internal authentication, private
  SHA-256 thread namespaces, allowlisted tools, approval pause and idempotency.
- Bot notification supports Redis idempotency.
- The separate document-upload path has a 50 MiB cap; upload filename/category
  handling resists traversal and writes through temporary file plus atomic
  replace.
- CHM content is not extracted onto the filesystem and has a per-member bound.
- External RAG transfer defaults to denied unless an explicit consent flag is
  set.
- `.dockerignore` excludes `.env`; the Docker runtime is non-root.
- Redis and Postgres are not published; Qdrant is bound to loopback.
- Telegram HTML is escaped before delivery.
- Ordinary LLM log previews use PII redaction and API keys use `SecretStr`.
- No direct `eval`, `exec`, shell execution or caller-controlled SSRF URL was
  found in runtime routes.

## Unverified areas

- Actual firewall/TLS/reverse-proxy rules, host port reachability and Phoenix
  authentication were not tested.
- Actual deployment token values were not read. The existing safe status script
  does not currently verify `INTERNAL_TOKEN` or `ADMIN_TOKEN`.
- Multi-replica broadcast/notification races and retention/encryption controls
  were not exercised.
- External OpenAI, Telegram, Qdrant, Postgres and Phoenix paths remain among the
  nine skipped integrations in the full suite.
- Dependency advisories are intentionally left to the dependency report.

## Verification evidence

- Focused document upload, media, bot notify, AI security and standalone API
  suites: **22 passed**.
- Focused observability, notifier, bot config and admin feedback suites:
  **16 passed**.
- An offline split-canary/split-email probe reproduced S-06.
- A broader route suite could not collect three modules because
  `langchain_openai` was absent from that local interpreter. This is an
  environment limitation, not three product test failures.
- `scripts/check_env_safety.py` was used without printing secret values. No
  `.env` contents or `*_KEY`, `*_TOKEN`, `*_SECRET` values were read.
- No external-network or mutating security test was performed.

## Formal verdict

**FAIL.** All five P1 issues are independently release-blocking. In particular,
the service cannot rely on an undocumented trusted network while Compose
publishes unauthenticated, billable, mutating and privacy-sensitive interfaces.
