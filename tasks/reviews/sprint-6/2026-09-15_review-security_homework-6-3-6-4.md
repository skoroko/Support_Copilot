# Security review: homework 6.3-6.4

Date: 2026-09-15

## Trust boundary

An internal caller authenticates to `POST /agent/stream`; the router validates
and namespaces the request before it reaches a shared persistent LangGraph.
The graph may call OpenAI and local safe tools. Telegram delivery crosses a
second authenticated app-to-bot boundary and is deduplicated in Redis.

## Resolved vulnerabilities

- The unauthenticated endpoint and client-selected `full` HIL bypass were
  closed with constant-time internal-token validation and an additional admin
  gate for `full`.
- Client thread IDs are namespaced by a SHA-256 digest of authenticated caller
  identity and client thread ID, preventing cross-identity resume/IDOR under
  the stated trusted-gateway model.
- Direct system/assistant/tool-message injection was closed with strict,
  bounded user-only request models and `input XOR resume` validation.
- SSE errors no longer expose provider exception details.
- Deterministic delivery keys, Redis `SET NX`, AOF and explicit
  `pending/sent/unknown` states block concurrent and ambiguous automatic
  resend.

## Open risks

- `X-User-ID` is asserted by the trusted internal gateway, not independently
  authenticated by this endpoint. The internal token must not reach ordinary
  clients or differently trusted services.
- Per-request bounds exist, but production ingress still needs per-identity
  rate/concurrency/token budgets and checkpoint retention.
- Checkpoint and tracing stores may retain support content; production needs a
  documented encryption, access-control, redaction and retention policy.
- Placeholder deployment credentials outside this route remain an operational
  configuration risk.

## Good controls

Dangerous tools fail closed, interrupt occurs before the external effect,
unknown tools cannot reach the notifier, request bodies are not logged, secret
values are not checkpointed, and all delivery tests use doubles.

## Unverified surfaces

- Reverse-proxy/WAF policy and Phoenix access control.
- High-concurrency resume and crash recovery across multiple replicas.

## Verdict

**PASS_WITH_WARNINGS** — no reachable P0/P1 exploit remains under the declared
trusted-gateway model.
