# Sealed holdout binding workflow

This workflow is offline. It never calls a model or provider and never modifies
sealed cases, questions, contexts, or the candidate index. Default execution reads
only `manifest.json` and `holdout_manifest.json`.

```powershell
& 'C:/00.MIDAS/PROJECTS/19. HelpDesk_Assistant/HelpDesk_Assistant/.venv/Scripts/python.exe' scripts/audit_holdout_references.py
```

After root explicitly authorizes final unlock, freeze the suite manifest and
actual candidate catalog hashes. Run from the integration checkout after copying
the script, or give absolute `--suite` and `--output` paths in the own checkout.

```powershell
& 'C:/00.MIDAS/PROJECTS/19. HelpDesk_Assistant/HelpDesk_Assistant/.venv/Scripts/python.exe' scripts/audit_holdout_references.py `
  --suite tests/eval/corpus_suite `
  --unlock-final-holdout `
  --root-freeze-id FINAL_FREEZE_ID `
  --sealed-manifest-sha256 FROZEN_SUITE_MANIFEST_SHA256 `
  --catalog var/s0005_acceptance/index/catalog.json `
  --catalog-sha256 FROZEN_ACTUAL_CATALOG_SHA256 `
  --source-root data `
  --output var/s0005_acceptance/holdout_reference_audit_v1
```

The first unlocked pass creates:

- `holdout_binding_contract.json`: source-byte SHA checks, whole sealed anchor
  matches with exact location and block fragments, pending causes, and original
  case fingerprints. It cannot invent an answer from an anchor.
- `holdout_candidate_contexts.json`: full native/reviewed text at exactly located
  source locations; fingerprints and block IDs. Unresolved visuals remain listed.
- `manifest.json`: input/output hashes, root freeze ID, implementation hash and
  counts. Existing output directories are rejected.

Matching preserves case and permits whitespace normalization only. Whole sealed
quotes must match; shortened/fuzzy anchors, filename aliases, guessed CHM topics,
and unproven duplicate representations do not satisfy binding. Source hash,
physical PDF page, or unique literal CHM location must be proven. Unknown source
version does not authorize cross-product/version applicability.

All initial candidates remain pending. A separate reader may provide a JSON file
with a `cases` list after reading the unlocked question and original material.
Each review needs these fields:

```text
case_id
case_sha256                     canonical digest of unchanged original case
catalog_sha256                  exact frozen catalog file SHA
sealed_cases_file_sha256        exact original holdout_cases.json file SHA
method = machine_source_review
human_expert_approval = false
reviewer, review_id, reviewed_at independent reader identity; never builder ID
acceptance_status = pending_independent_review
reference_answer                actual query-relative answer, never anchor proxy
query_relative_coverage_verified = true
sufficiency_rationale           how the evidence answers every requested part
applicability_status = source_specific_supported
required_visual_status = not_required | approved_literal_bound
claim_slots = [{slot_id, claim, alternatives:[{evidence:[...]}]}]
```

Each exact evidence record needs `source_hash`, `relative_path`, `location_id`,
`block_id`, physical `page` or null, exact `chm_member` or null, literal `quote`
and canonical `quote_sha256`. Reviewed visual evidence also needs `source_review`
equal to the entire actual block attributes; the actual catalog must mark it
verified and retain its independent literal review ID. Required visual regions
need `required_visual_evidence` mapping each `slot_id:region_name` to approved
visual evidence block IDs. Every originally required source must be represented.
No source/group substitutions are accepted.

No-answer/clarification cases also require a separately reviewed behavioral
answer; a retrieval miss cannot establish corpus absence. The CLI does not use
provider outputs, lexical proxies, or regex approval to establish semantics.

Run a new output version with `--machine-reviews PATH`. Only valid separately
reviewed candidates receive `machine_reviewed_sufficient_candidate`; this records
the reader's declared source-specific sufficiency and proves mechanical evidence
binding. Root's independent semantic reference acceptance remains required. The
builder cannot certify that a declared review is semantically correct.

Invalid review, visual, applicability, layout or source gaps stay pending with a
reason. All original questions and references remain unchanged. Runtime retrieval
recall over this contract is a mechanical evidence coverage measurement; release
still requires accepted references and all required layers, groups and latency
gates. This script is a standalone audit builder, not a replacement runtime loader.

Tests use synthetic cases only and never open project holdout content.
