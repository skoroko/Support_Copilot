# Dependency review: homework 6.3-6.4

Date: 2026-09-15

## Evidence

- `pip check`: no broken requirements.
- Installed LangGraph/LangChain, SQLite/Postgres saver, `aiosqlite` and psycopg
  versions satisfy the declared ranges.
- Import smoke for both checkpoint backends passed.
- No Git, local-file, tarball or custom-index sources were found in requirement
  manifests.
- `docker compose config --quiet` passed; the optional Qdrant key warning does
  not invalidate the graph services.

## Warnings

- There is no dependency lockfile; broad lower-bound ranges reduce clean-build
  reproducibility.
- `pyproject.toml` declares no runtime dependencies while Docker installs
  `requirements.txt`, so the two manifest surfaces can drift.
- `arizephoenix/phoenix:latest` and base images are mutable rather than pinned
  to immutable digests.
- Direct `aiosqlite`/psycopg declarations partly overlap saver transitives but
  intentionally document supported backends and psycopg extras.

## Unverified areas

`pip-audit`/`safety` are not installed and the registry audit could not reach
PyPI in the restricted environment. Current CVE and registry provenance status
therefore remains unverified.

## Verdict

**PASS_WITH_WARNINGS** — dependency resolution is currently consistent; the
warnings concern reproducibility and unavailable vulnerability-audit evidence.
