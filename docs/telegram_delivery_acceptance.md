# S0-005: isolated candidate and Telegram delivery measurement

This is an acceptance tool, not a production rollout. Production containers,
image tag `llm-service:v1`, `.env`, database, Redis and Qdrant collection remain
untouched. `compose.s0005.yaml` is a standalone Compose project; never merge it
with the working deployment's Compose files.

Candidate API binds `127.0.0.1:18001`; its PostgreSQL/checkpoint and Redis use new
project volumes without host ports. Qdrant is the separate acceptance instance
on port 16333, collection `s0005_candidate_v1`. Candidate data mounts are read
only. Its output/index and shared budget SQLite ledger live in a separate
acceptance directory. No bot polling, webhook, broadcast worker or notify server
starts. Outbound action transport is disconnected; action approval buttons are
delivered but never clicked by the harness.

The provisional frozen candidate uses `gpt-6-luna`, Responses API, low reasoning
and 500 output tokens for brief answers, medium and 1500 for expanded answers,
with an 8192 reasoning reserve. The working runtime profile is unchanged.
This harness selects no new models/prices. The acceptance budget adapters installed before startup
account for provider calls in one durable ledger with the task's $10 ceiling.
Uncertain calls keep their full reservation. Do not reset the ledger between
indexing, retrieval, Telegram scenarios or expanded requests.

Before launch, the operator supplies process variables `S0005_SECRETS_FILE`
(existing private file, never printed), `S0005_DATA_DIR`, `S0005_ARTIFACT_DIR`
and `S0005_CODE_DIR` (frozen candidate source). Build only the unique candidate
image tag, or use the already built candidate image and the read-only app/bot
source mounts. Create the candidate index and shared ledger first. Use
`docker compose -f compose.s0005.yaml up -d`; do not invoke production compose.

The authenticated `/__candidate_identity` endpoint returns source byte hashes,
allowlisted runtime profile, index/catalog hashes and shared budget summary.
It exposes no credential/recipient values or source text. Freeze those bytes
before measuring. The harness checks the expected aggregate identity and that
its local bot handlers exactly match the candidate backend source hash.

Run `scripts/evaluate_telegram_delivery.py --help` for the executable arguments.
One invocation handles one non-command text question in `--question-file` and
requires a fresh `--output`, explicit `--authorized-working-chat`, the confirmed
`--recipient-fingerprint`, and `--candidate-identity`. Recipient resolution
requires exactly one positive allowed user ID and a matching SHA256. It reads
the ID privately from BotSettings; IDs and tokens never enter reports or stdout.
Use the project Python 3.13 environment. The default URL is strictly restricted
to the candidate loopback port 18001.

The harness injects a synthetic incoming Update into the unchanged Dispatcher,
AdmissionMiddleware and text/burst handler. Timing begins immediately before
dispatch, so normal text debounce is included. It calls the candidate backend
and the real Telegram Bot API, awaiting every message/edit ACK, including all
split answer chunks and final inline controls. It then exercises the existing
documents handler and awaits the actual documents message ACK. The report
distinguishes first body ACK, final answer ACK, and final delivery ACK including
documents. HTTP completion is never used as delivery completion.

Inbound Telegram polling and genuine Telegram callback ACK are not measured:
callbacks are injected in process, with callback acknowledgement explicitly
labelled unmeasured. Source panel delivery still uses real Telegram sends.
`--confirm-expanded` explicitly opts into the existing expansion cost notice
and owner-bound confirmation handlers and an additional billed request. Brief
measurement is the default. The harness never approves an external action.

Reports contain timestamps, hashes, text lengths, button kinds, safe exception
types/HTTP statuses and before/after shared budget summaries. A separate private
`.canonical.json` sidecar retains the persisted brief/expanded answers and sources
for semantic review, bound to the report by SHA-256. Neither file contains raw
recipient IDs, callback payloads, credentials or exception bodies. The candidate's
separate chat history is cleared and verified empty before the receipt timer;
working history and visible Telegram messages are untouched. Expanded latency
is also measured separately from explicit confirmation to the last body ACK.
An in-flight checkpoint is saved before dispatch; an existing result is never
automatically replayed, including failed or uncertain attempts. A failed HTTP
request, transport exception, missing canonical backend answer, missing controls,
or failed documents delivery is retained as a failed measurement.

Offline tests use a fake Telegram session and httpx MockTransport with the real
dispatcher/burst/SSE/rendering/source handlers. Those tests validate measurement
contracts, not actual Telegram network delivery. Real delivery results require
running the authorized harness separately and retaining its report and ledger.
