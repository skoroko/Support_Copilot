"""Bound independent recall operations and preserve order and partial failures."""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchOutcome:
    query: str
    items: list[Any]
    elapsed_ms: float
    error_type: str | None = None


def independent_searches(
    queries: Sequence[str], search: Callable[[str], Sequence[Any]], *,
    concurrency: int = 2, deadline: float | None = None,
) -> list[SearchOutcome]:
    """Join bounded workers, retaining successful siblings after a failure.

    The search adapter must enforce its own provider timeout. Joining workers
    deliberately prevents detached provider calls escaping request accounting.
    Error messages and query contents are never logged here.
    """
    def one(query: str) -> SearchOutcome:
        started = time.perf_counter()
        try:
            if deadline is not None and started >= deadline:
                raise TimeoutError("Retrieval deadline exhausted")
            items = list(search(query))
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Retrieval deadline exhausted")
            return SearchOutcome(query, items, (time.perf_counter() - started) * 1000)
        except Exception as exc:  # noqa: BLE001 - error bodies can contain credentials
            return SearchOutcome(query, [], (time.perf_counter() - started) * 1000, type(exc).__name__)

    workers = min(max(1, concurrency), len(queries), 4)
    if workers <= 1:
        return [one(query) for query in queries]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="corpus-recall") as executor:
        futures = [executor.submit(copy_context().run, one, query) for query in queries]
        return [future.result() for future in futures]


def search_while_translating(search: Callable[[], Any], translate: Callable[[], str | None]):
    """Start original-language recall immediately; join before English recall.

    Translation remains auxiliary and never holds up starting the original
    searches. Both jobs inherit the turn deadline. No worker escapes the join.
    """
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="original-recall") as executor:
        original = executor.submit(copy_context().run, search)
        try:
            translated = translate()
        except Exception:  # Translation failures must retain original recall.
            translated = None
        return original.result(), translated
