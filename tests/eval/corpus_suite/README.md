# Manual corpus evaluation candidates

120 independent question blueprints: 90 development and 30 sealed holdout.
Legacy 34 golden + 9 additional turns + 5 expanded modes remain separate and
unchanged. All nine allowed PDFs are covered. CHM and derived HTML share one
parent source; they are not independent documentation authorities.

**Every reference is pending_review.** Exact quote matching establishes location
and integrity, not answer correctness, formula completeness, source-version
applicability or expert endorsement. Questions involving multiple documents are
candidate synthesis contracts; an engineer must approve or reject applicability.
This suite cannot currently authorize release. Known embedded formula/table
losses have required visual-region identifiers; a text match cannot satisfy them.

The default runner reads only dev cases and dev contexts. Holdout content must
not be provided to tuning agents. Final evaluation requires both `--split holdout`
and `--unlock-holdout-final-evaluation`. Its questions, source records and
references must stay out of development reports. Explicit object/section topic
tags also have zero overlap across splits; these tags do not prove complete
semantic independence, which still requires an expert split review. Do not regenerate or repartition
the sealed suite after tuning. A changed dataset requires a new version/directory.

## Offline commands

Run from the documentation-eval checkout with the existing managed interpreter:

```powershell
$taskPython = 'C:/00.MIDAS/PROJECTS/19. HelpDesk_Assistant/HelpDesk_Assistant/.venv/Scripts/python.exe'
& $taskPython scripts/evaluate_corpus_suite.py --output tests/eval/corpus_suite/runs/new-dryrun.json
& $taskPython scripts/evaluate_corpus_suite.py --run-extraction --source-root 'C:/00.MIDAS/PROJECTS/19. HelpDesk_Assistant/HelpDesk_Assistant' --output tests/eval/corpus_suite/runs/new-extraction.json
& $taskPython scripts/evaluate_corpus_suite.py --catalog-retrieval-smoke PATH-TO-FROZEN-CATALOG.json --output tests/eval/corpus_suite/runs/new-lexical-smoke.json
& $taskPython -m pytest tests/unit/test_corpus_suite.py -q --basetemp=.test-tmp-corpus-suite-new
```

Each run path must be new. A dry run produces explicit `not_run` attempts; it
never constructs providers. Extraction hashes all 70 files, verifies the frozen
docstore identity, and re-extracts PDF pages / HTML / CHM independently. Its
reader acquisition is cached and reported separately; those timings are not
production ingestion throughput or e2e response latency. No-answer extraction
checks integrity only; corpus absence remains an expert semantic judgment.
All source defaults are portable relative to this checkout. `--source-root`
explicitly selects a separate original checkout without copying/mutating its
manuals, index or docstore. The sealed manifest retains original file hashes.

The optional catalog smoke reads one frozen catalog byte snapshot, fingerprints
it and validates source hashes against sealed anchors. It ranks literal catalog
blocks by local token overlap, retaining exact block/location/source identities.
`candidate_block_recall_at_10` uses mechanically located exact 120-character
anchor spans on the same source/page; candidate targets stay pending semantic
review. `evidence_recall_at_10` retains the stricter entire required-quote match.
Both metrics expose their denominators and neither is an approved relevance gold.
It performs no embeddings, translation, providers or network calls. Its recall
is a weak lexical diagnostic: an anchor may span multiple blocks, and Russian
queries may not match English evidence. Smoke results explicitly block production
retrieval/semantic release even if their exact-anchor recall is high.

## Explicit live execution (not run by this implementation task)

```powershell
& $taskPython scripts/evaluate_corpus_suite.py --live --allow-external-data-transfer --layers retrieval,generation --repeats 3 --output NEW-LIVE.json
& $taskPython scripts/evaluate_corpus_suite.py --live --allow-external-data-transfer --enable-e2e --layers e2e --api-url http://127.0.0.1:18000 --output NEW-E2E.json
```

Live retrieval disables ingestion/rebuilding. Fixed generation uses AgentModel
with identical sealed contexts for every profile; references are never sent to
the answering model. `--profiles` accepts JSON `label -> {model, effort, api}`
without changing runtime defaults. E2E reuses BackendClient auth, creates
isolated evaluation chats, checks canonical persisted text/sources and performs
no Telegram delivery. It only measures the configured deployed model, so custom
generation profiles are rejected for that layer. The first HTTP content event is
not model TTFT. Delivery latency, unmeasured usage/cost and TTFT remain null.

Independent layers may be run separately for diagnostics. The strict release
gate requires a complete combined run, expert references and reviews, all five
source groups, all four layers, repeated observations, holdout and independently
verified legacy regression. It deliberately stays blocked when these are absent.
No run-merging or automatic legacy approval is implemented; the
`legacy_regression_verified` field starts false and must not be toggled manually.

## Expert ledgers

To approve a candidate reference, supply `--reference-reviews` with:

```json
{"suite_manifest_sha256":"canonical manifest digest","references":[
  {"case_id":"dev case ID","case_sha256":"canonical sealed case digest",
   "method":"expert","status":"approved","reviewer":"engineer identity",
   "notes":"source/version/applicability and completeness assessment",
   "reference":"reviewed answer contract","applicability_checked":true}
]}
```

Approval is applied in memory; sealed questions/contexts are not rewritten.
Reference and result review ledgers are distinct. Result review requires the
saved run's `run_id` and `binding_sha256`, plus the exact raw attempt fingerprint:

```json
{"run_id":"saved run","run_binding_sha256":"saved binding","reviews":[
  {"attempt_id":"profile:layer:case:repeat","raw_attempt_sha256":"raw_row_sha(row)",
   "method":"expert","reviewer":"engineer identity","notes":"cited-fragment assessment",
   "correctness":1,"completeness":1,"grounding":1,"sources_valid":true,"critical_error":false}
]}
```

Use `scripts/evaluate_corpus_suite.py --score-run SAVED.json --expert-review REVIEW.json
--reference-reviews REFERENCES.json --output NEW-REVIEWED.json`. Scores are only
0/0.5/1; failed attempts require correctness=0 and grounding=null. No regex,
keyword inclusion, citation marker or filename match approves a semantic answer.
Missing reviews remain unknown and block release. All failures count in combined
quality/source/15-second denominators and all-attempt p50/p95; success-only timing
is separately labeled. Group/layer/profile and task-type denominators are explicit.

Optional `--visual-transcriptions` accepts the shared corpus module's records:
`source_hash`, physical `page`, top-down PDF-point `bbox`, decoded-image
`image_sha256`, exact `product`/`version`, `text`, `kind`,
`approval_status=approved`, `reviewer`, `reviewed_at`, `review_id`.
`CorpusCatalog.build(..., transcriptions=records)` reuses
`apply_reviewed_transcriptions` to reject stale/ambiguous images. Evaluator labels
only describe missing content; they cannot approve a region. Until an expert
selects specific required bboxes the conservative check requires all unresolved
images on the flagged page to have verified source-bound transcriptions.
Production use of that content must still be demonstrated by
retrieval/generation/e2e evidence, not inferred from a sidecar.
