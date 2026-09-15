# Quality review: homework 6.3-6.4

Date: 2026-09-15
Scope: the accumulated homework 6.3 and 6.4 diff, with emphasis on the
persistent graph, HIL transition, SSE transport and delivery boundary.

## Confirmed findings and corrections

1. Parallel tool calls initially left unmatched provider messages. The graph
   now handles every latest call, including mixed safe/dangerous batches.
2. A tool call produced on the sixth model turn was initially left unmatched.
   `finish` now emits a rejection `ToolMessage` before the escalation message.
3. SSE provider failures initially ended an HTTP 200 stream without a terminal
   event. They now emit a constant `agent_stream_failed` event without internal
   exception text.
4. Telegram delivery initially had an ambiguous retry window. A deterministic
   action key now flows through the notifier into Redis states
   `pending -> sent|unknown`; unknown outcomes block automatic resend. The
   notifier rejects both non-2xx and false-success response bodies, and the
   graph leaves `sent=False` on failure.

All findings were reproduced before correction and covered by regression
tests afterwards.

## Unverified areas

- Exactly-once delivery is not promised by Telegram; the chosen contract is
  at-most-once with explicit reconciliation for `unknown` outcomes.
- Load/concurrency testing of Postgres checkpoints and Redis delivery claims
  remains outside this homework scope.

## Verdict

**PASS** — no confirmed quality blocker remains.
