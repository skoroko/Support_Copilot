"""Budgeted literal previews of already validated tables, for navigation only.

The caller owns DOM/source/scope validation and revalidation after selection.
Previews establish no header semantics, completeness or citable evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

_CELL = re.compile(r"(?:^| \| )Column \d+=")


class TableNavigationInputError(ValueError):
    """All candidate identities and minimum previews cannot fit safely."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _examples(lines: list[str]) -> list[str]:
    """One literal cell per following row; exact repetitions rank last.

    This is mechanical duplicate handling, not a judgement about which column
    is a model/category. Values unique across the three rows are considered
    first; no value is invented or normalized beyond taking a literal excerpt.
    """
    rows = [[part.strip() for part in _CELL.split(line) if part.strip()] for line in lines[1:4]]
    counts = Counter(value for row in rows for value in set(row))
    result = []
    for row in rows:
        values = sorted(enumerate(row), key=lambda pair: (counts[pair[1]] != 1, pair[0]))
        value = next((value for _, value in values if value not in result), None)
        if value is not None:
            result.append(value)
    return result


def _preview(base: dict[str, Any], values: list[str], cap: int) -> dict[str, Any]:
    result = {**base, "examples": [value[:cap] for value in values] if cap else []}
    partial = [index for index, value in enumerate(values) if len(value) > cap] if cap else []
    if partial:
        result["partial_examples"] = partial
    return result


def _with_examples(base: dict[str, Any], values: list[str], allowance: int, max_description_chars: int) -> str:
    baseline = _json(base)
    base_cost = len(_json(baseline))

    def fits(cap: int) -> bool:
        description = _json(_preview(base, values, cap))
        return len(description) <= max_description_chars and len(_json(description)) - base_cost <= allowance

    if not values:
        return baseline
    maximum = min(max(map(len, values)), max_description_chars)
    # Removing a partial marker can reduce serialized length. Search each
    # interval with a fixed partial set separately, rather than assuming global
    # monotonicity. There are at most three example lengths/interval boundaries.
    boundaries = sorted({1, maximum + 1, *(len(value) for value in values if len(value) <= maximum)})
    best = 0
    for start, stop in pairwise(boundaries):
        low, high = start, stop - 1
        if not fits(low):
            continue
        while low < high:
            middle = (low + high + 1) // 2
            if fits(middle):
                low = middle
            else:
                high = middle - 1
        best = max(best, low)
    return _json(_preview(base, values, best))


def build_table_candidates(
    question: str, groups: Sequence[Mapping[str, Any]], *, max_input_chars: int = 40_000,
    max_description_chars: int = 1_500,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Preserve every supplied ID/order and fit the selector's shared JSON.

    Reserve the question, identities, scope and compact title/row-zero prefixes
    first, then share the remaining serialized-character budget equally among
    every candidate. Whole example values are preferred; partial value indices
    are explicit. No source text is changed and no candidates are removed.
    Oversized irreducible inputs fail explicitly; callers must skip selection.
    """
    if (not isinstance(question, str) or not question.strip() or len(question) > 40_000
            or not isinstance(groups, Sequence) or isinstance(groups, (str, bytes))
            or not 1 <= len(groups) <= 100 or type(max_input_chars) is not int
            or not 1 <= max_input_chars <= 40_000 or type(max_description_chars) is not int
            or not 1 <= max_description_chars <= 1_500):
        raise TableNavigationInputError("input_bounds")
    prepared = []
    identities = set()
    for group in groups:
        if not isinstance(group, Mapping):
            raise TableNavigationInputError("group_contract")
        identity, text, meta = group.get("id"), group.get("text"), group.get("metadata")
        if (not isinstance(identity, str) or not identity.strip() or identity.casefold() == "none"
                or identity in identities or len(identity) > 40_000 or not isinstance(text, str)
                or not text.strip() or not isinstance(meta, Mapping)
                or any(not isinstance(meta.get(key), str) for key in ("product", "version"))
                or meta.get("title") is not None and not isinstance(meta["title"], str)):
            raise TableNavigationInputError("group_contract")
        identities.add(identity)
        # Rendered DOM rows start with Column 1=. Preserve literal embedded
        # newlines inside a cell instead of treating them as extra table rows.
        lines = re.split(r"\n(?=Column 1=)", text) if text.startswith("Column 1=") else text.splitlines()
        prepared.append((identity, meta, lines[0], _examples(lines)))

    # Prefixes remain literal. If the question leaves very little space, reduce
    # every prefix by the same policy, never by candidate rank or subject.
    for title_limit, row_limit in ((48, 64), (24, 32), (12, 16), (6, 8), (3, 4), (1, 1)):
        bases = [{"product": meta["product"], "version": meta["version"],
                  "title": meta.get("title", "")[:title_limit] if meta.get("title") is not None else None,
                  "row0": row[:row_limit], "examples": [], "preview": True,
                  "not_citable": True, "truncated": True}
                 for _, meta, row, _ in prepared]
        candidates = [{"id": identity, "description": _json(base)}
                      for (identity, _, _, _), base in zip(prepared, bases)]
        shared = _json({"question": question, "candidates": candidates})
        if (len(shared) <= max_input_chars
                and all(len(candidate["description"]) <= max_description_chars for candidate in candidates)):
            break
    else:
        raise TableNavigationInputError("irreducible_input_budget")

    allowance = (max_input_chars - len(shared)) // len(candidates)
    for candidate, base, (_, _, _, values) in zip(candidates, bases, prepared):
        candidate["description"] = _with_examples(base, values, allowance, max_description_chars)
    shared = _json({"question": question, "candidates": candidates})
    if len(shared) > max_input_chars or any(len(row["description"]) > max_description_chars for row in candidates):
        raise TableNavigationInputError("serialized_input_budget")
    descriptions = [json.loads(row["description"]) for row in candidates]
    return candidates, {
        "origin": "literal_table_navigation", "candidate_count": len(candidates),
        "input_chars": len(shared), "unused_input_chars": max_input_chars - len(shared),
        "input_sha256": hashlib.sha256(shared.encode("utf-8")).hexdigest(),
        "sampled_candidates": sum(bool(row["examples"]) for row in descriptions),
        "partial_example_count": sum(len(row.get("partial_examples", [])) for row in descriptions),
        "preview_only": True, "title_prefix_chars": title_limit, "row0_prefix_chars": row_limit,
    }
