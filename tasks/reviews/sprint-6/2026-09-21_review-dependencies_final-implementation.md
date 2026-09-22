# Dependency review: final implementation

Date: 2026-09-21
Baseline: `a519aa7b682b1872824f71e4b9e83db650065ed8`
Scope: Python runtime, tracing, reranker and evaluation manifests; resolved
local environments; Docker build inputs and Compose service images. Application
security and general code-quality findings are intentionally excluded.

## Result

**FAIL** — two confirmed P1 reproducibility/supply-chain blockers and one P1
unresolved licensing decision block release. Two P2 advisory risks and one P3
runtime hygiene issue remain.

| Severity | Count | Classification |
| --- | ---: | --- |
| P0 | 0 | — |
| P1 | 3 | 2 confirmed technical blockers; 1 unresolved release decision |
| P2 | 2 | advisory/risk findings with bounded or unproven production reachability |
| P3 | 1 | runtime hygiene |

## Confirmed findings and release gates

### D-01 — P1 — The Python release graph is not locked or hash-pinned

**Evidence.** `requirements.txt:1-53` primarily uses open lower bounds; only
the Qdrant/LlamaIndex integration boundary is exactly pinned.
`pyproject.toml:5` declares `dependencies = []`, while Docker installs the
separate requirements file. No `uv.lock`, compiled constraints, hash-locked
requirements, Poetry/Pipenv lock or `--require-hashes` release input was found.
`Dockerfile:21-28` resolves the graph afresh during each build. Two valid local
resolutions already differ: `.venv-eval` contains `alembic 1.19.2`, while the
running image contains `alembic 1.20.0`.

**Impact.** The same Git commit can produce different runtime graphs across
time and Python versions, silently introducing upgrades, incompatibilities or
new vulnerabilities. Git alone cannot reconstruct a known release artifact.

**Suggested action.** Commit Python-version-specific lock/constraints files
for runtime and each optional group, include transitive hashes, install the
release image only from the lock, and designate one authoritative dependency
declaration.

### D-02 — P1 — Container inputs are mutable tags rather than immutable digests

**Evidence.** `Dockerfile:3,37` uses mutable
`python:3.13-slim-bookworm`; `Dockerfile:5` uses a mutable version tag for
`ghcr.io/astral-sh/uv:0.6.10`. `compose.yaml:64,76,118,133` uses mutable tags
for service images, including `arizephoenix/phoenix:latest`. Locally resolved
digests exist, but none are preserved in the tracked release configuration.

**Impact.** Pulling or rebuilding an unchanged commit can replace the OS,
build tool, cache, database, vector store or observability runtime. Incident
reproduction and rollback are therefore not deterministic.

**Suggested action.** Pin each base and service image as
`tag@sha256:digest`, including Phoenix, Python, uv, Redis, Postgres and Qdrant;
update digests through a reviewed process and retain an image/SBOM identity per
release.

### D-03 — P1 release gate — PyMuPDF licensing has no recorded decision

**Evidence.** `requirements.txt:35` directly declares `pymupdf>=1.24.0`.
`app/services/ingestion.py:25,352` reaches it through LlamaIndex
`PyMuPDFReader` during PDF ingestion. Installed PyMuPDF 1.28.2 metadata reports
dual licensing under AGPL-3.0 or an Artifex commercial license. No repository
license policy, third-party notices, SBOM, documented AGPL compliance path or
evidence of an applicable commercial license was available to the review.

**Judgment boundary.** This is **not** a confirmed legal violation. It is a
release-blocking unresolved decision: the available evidence is insufficient
to establish which license basis applies to the intended proprietary/networked
deployment.

**Impact.** Releasing without the decision can create unassessed source-
disclosure or commercial-license obligations.

**Suggested action.** Obtain and record an owner/legal decision: document the
AGPL compliance basis, record the applicable commercial license, or replace
the reader with a dependency compatible with the approved policy. Add license
inventory/SBOM generation to the release gate.

## Advisory and hygiene findings

### D-04 — P2 — The runtime graph contains a high-severity NLTK advisory with unproven reachability

**Evidence.** The authoritative dependency audit of `requirements.txt`
resolved `nltk 3.10.3` and reported `PYSEC-2026-3740` /
`GHSA-8mgp-746c-j5xp` (`CVE-2026-81726`), affected through 3.10.3 with no fixed
version published at review time. The chain is
`llama-index -> llama-index-core -> nltk>=3.9.3`. Static inspection found no
direct use of the affected `TransitionParser`, `AveragedPerceptron`,
`PerceptronTagger` or `save_maxent_params` paths.

**Reachability judgment.** The vulnerable package is shipped, but an
application-level exploit path was not demonstrated. This is therefore P2,
not a confirmed production exploit.

**Suggested action.** Record a time-bounded risk acceptance, forbid affected
APIs with untrusted artifact paths, and pin the first patched release or remove
the transitive dependency when feasible.

### D-05 — P2 — The optional evaluation environment resolves known-vulnerable packages

**Evidence.** The audit of `requirements-eval.txt` resolved advisory records
for four packages: `strawberry-graphql 0.270.1` (fixed by 0.312.3/0.315.7),
`langchain-openai 1.1.9` (fixed by 1.1.14), `ragas 0.4.3` (no fixed version
listed), and `diskcache 5.6.3` (no fixed version listed). Duplicate database
rows were not counted as distinct findings. `requirements-eval.txt:10` pins
the older Phoenix 11.38.0 evaluation stack. `Dockerfile:16-28` does not install
this optional group in the default production image.

**Impact.** The default release image is not exposed through this group, but
local evaluation/Phoenix workflows can execute the affected stack and may
process production-like support data.

**Suggested action.** Lock and upgrade the evaluation environment separately,
or isolate it from production data and network exposure while advisories
without fixes remain.

### D-06 — P3 — Test tooling is installed in the production image

**Evidence.** `requirements.txt:20-21` includes `pytest` and
`pytest-asyncio`; `Dockerfile:22` installs the whole file into the virtualenv
copied into the final image. Production modules do not import either package.

**Impact.** The image carries unnecessary size and attack surface, and the
runtime/development dependency boundary is obscured.

**Suggested action.** Move test tooling to a dev/test lock or optional group
and exclude it from the release image.

## Good controls observed

- `pip check` reported no broken requirements in `.venv-eval`; the dependency
  agent also confirmed the currently running production image is resolver-
  consistent.
- `docker compose config --quiet` passed. The warning about an unset local
  `QDRANT_API_KEY` is expected local configuration, not invalid Compose syntax.
- No Git, `file:`, direct tarball or direct URL Python requirements were found.
- The tracing and reranker requirement groups produced no known advisory in
  the performed audit.
- Heavy reranker dependencies are kept out of the default image.
- The Docker build is multi-stage and the runtime uses non-root `appuser`.
- Exact Qdrant/LlamaIndex integration pins reduce drift at that specific
  compatibility boundary.

## Questions and unverified areas

- The organization license policy and evidence of an Artifex commercial
  license were unavailable; D-03 requires owner/legal confirmation.
- OS/base-image CVEs were not scanned with Trivy, Grype or Docker Scout.
- NLTK reachability was assessed statically; no reachability scanner or
  exploit test was run.
- Python `>=3.10` is declared while the release image uses 3.13. Without
  per-version locks and a CI matrix, lower-bound compatibility is unproven.
- No SBOM, provenance attestation, continuous license/vulnerability gate or
  dependency-update automation was found.
- Registry availability or a local audit-tool limitation would be an
  environment limitation; the findings above rely on the completed dependency
  agent audit and tracked manifests, not on treating network unavailability as
  a product defect.

## Verification evidence

- `.venv-eval`: `pip check` passed.
- Running production image: dependency-agent `pip check` passed.
- `docker compose config --quiet`: passed, with only the expected local
  Qdrant-key warning.
- Public advisory audit: runtime NLTK risk and four affected optional-eval
  packages recorded above; tracing and reranker groups had no reported known
  advisory.
- No network or package mutation was performed by the final aggregate pass.

## Formal verdict

**FAIL.** Release and merge approval are withheld until D-01 and D-02 are
resolved and D-03 has an explicit approved licensing disposition. D-03 must
not be represented as proof of infringement; it is a mandatory unresolved
release decision.
