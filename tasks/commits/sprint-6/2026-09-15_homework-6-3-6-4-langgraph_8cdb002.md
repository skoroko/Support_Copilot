# Commit report: 8cdb002

## Summary

Commit `8cdb002` (`feat: add persistent LangGraph agents`) completes homework
6.3 and 6.4 as one cumulative Support Copilot increment.

Linked task/issue: local course assignments `self-practice-6-3` and
`self-practice-6-4`; no GitHub issue was linked.

## Affected files

- Agent domain: `app/services/agent_graph.py`,
  `app/services/agent_persistent.py`.
- API/lifecycle: `app/routers/agent.py`, `app/main.py`.
- Delivery boundary: `app/services/notifier.py`, `bot/web.py`,
  `bot/config.py`, `bot/__main__.py`.
- Persistence/deployment: `alembic/env.py`, `compose.yaml`, `Dockerfile`,
  `.dockerignore`, `.env.example`, `requirements.txt`.
- Demonstration/evaluation: `scripts/bench_agents.py`,
  `scripts/time_travel_demo.py`, `scripts/visualize_graph.py`.
- Tests: `tests/test_agent_graph.py`, `tests/test_agent_persistent.py`,
  `tests/app/test_notifier.py`, `tests/bot/test_web.py`.
- Documentation: homework README/checklists, Mermaid diagrams, implementation
  reports and sprint-6 review reports.

## Behavioral details

- Adds bounded custom and prebuilt LangGraph agents with shared support tools.
- Adds memory, SQLite and Postgres checkpoint backends with one-time lifespan
  setup and stable private thread lineage.
- Adds authenticated SSE updates/messages and HIL approve/reject resume.
- Rejects non-user message injection and limits one request to four bounded
  messages.
- Pairs every provider tool call, including mixed calls and the sixth-turn
  safety cutoff.
- Adds deterministic Telegram action IDs and Redis/AOF
  `pending/sent/unknown` deduplication. Unknown delivery outcomes are not
  automatically retried or falsely marked sent.

## Decisions and documentation impact

LangGraph owns its four checkpoint tables; Alembic explicitly excludes them.
The existing Postgres and Redis services are reused. The HTTP trust model is an
authenticated internal gateway asserting `X-User-ID`; the privileged `full`
role additionally requires the administrative credential.

## Verify

- `.venv-eval\Scripts\python.exe -B -m pytest -q` — PASS: 291 passed,
  9 skipped.
- Targeted persistent/notifier/bot tests — PASS: 22 passed.
- `.venv-eval\Scripts\python.exe -B scripts\time_travel_demo.py` — PASS:
  interrupt, history, past state, reject and approve branches.
- `python -m py_compile ...` for changed Python modules — PASS.
- `.venv-eval\Scripts\python.exe -m pip check` — PASS.
- `docker compose config --quiet` — PASS.
- `docker compose up -d --build app bot` — PASS.
- Authenticated `POST /agent/stream` live smoke — PASS: HTTP 200, terminal
  `done`, zero tool calls and no Telegram delivery.
- Docker app/Postgres/Redis checks — PASS: healthy services, Redis AOF enabled,
  four checkpoint tables present.
- `git diff --cached --check` — PASS.

## Breaking changes and migrations

`POST /agent/stream` is a new endpoint and requires internal authentication;
there is no previous public contract to migrate. The bot notify payload accepts
an optional idempotency key, so existing callers remain compatible. No Alembic
migration is introduced; checkpoint schemas are created by the saver.

## Remaining risks

- Production still needs per-identity rate/concurrency limits and checkpoint/
  trace retention policy.
- `X-User-ID` is trusted only because the internal token is restricted to a
  trusted gateway.
- Dependency builds are not lockfile-reproducible and registry CVE auditing was
  unavailable in the restricted environment.
- Telegram `unknown` outcomes require operator reconciliation by design.
