"""Literal previews preserve all identities and the real selector input bound."""
import hashlib
import json
from copy import deepcopy

import pytest

from app.services import table_navigation
from app.services.table_navigation import (
    TableNavigationInputError,
    build_table_candidates,
)
from app.services.table_selection import TableSelectionService


def group(identity="a", *, title="Literal choices", values=("Alpha", "Beta", "Gamma")):
    rows = ["Column 1=Category | Column 2=Choice | Column 3=Flag"]
    rows.extend(f"Column 1=Repeated type | Column 2={value} | Column 3=O" for value in values)
    return {"id": identity, "anchor_text": "Literal metadata", "text": "\n".join(rows),
            "metadata": {"product": "fictional", "version": "unknown", "title": title}}


def descriptions(candidates):
    return [json.loads(row["description"]) for row in candidates]


def test_distinct_literal_cells_from_three_rows_and_non_citable_markers():
    groups = [group("material", title="Material", values=("Elastic (Упругая)", "Tresca (Треска", "von Mises")),
              group("property", title="Property", values=("Geogrid(1D)", "Plot only(1D)", "Truss (Стержни)"))]
    original = deepcopy(groups)
    candidates, report = build_table_candidates("Which choices apply?", groups)
    material, properties = descriptions(candidates)
    assert material["examples"] == ["Elastic (Упругая)", "Tresca (Треска", "von Mises"]
    assert properties["examples"] == ["Geogrid(1D)", "Plot only(1D)", "Truss (Стержни)"]
    assert all(row["preview"] and row["not_citable"] and row["truncated"] for row in (material, properties))
    assert report["partial_example_count"] == 0 and report["sampled_candidates"] == 2
    assert groups == original


def test_100_candidates_preserve_ids_order_and_every_candidate_gets_examples():
    groups = [group(f"{index:064x}", values=(f"Choice {index} A", f"Choice {index} B", f"Choice {index} C"))
              for index in range(100)]
    candidates, report = build_table_candidates("Перечисли варианты / list available choices.", groups)
    assert [row["id"] for row in candidates] == [row["id"] for row in groups]
    shared, _ = TableSelectionService._input("Перечисли варианты / list available choices.", candidates)
    assert len(shared) <= 40_000 and max(len(row["description"]) for row in candidates) <= 1500
    assert report["input_chars"] == len(shared)
    assert report["input_sha256"] == hashlib.sha256(shared.encode()).hexdigest()
    assert report["sampled_candidates"] == 100


def test_long_unicode_and_control_cells_are_explicit_literal_excerpts():
    values = ('α θ "quoted" \\ Ω ' * 100, 'Кириллица\t±β\n ' * 100, '长表格 Ω ' * 100)
    groups = [group(f"{index:064x}", values=values) for index in range(100)]
    candidates, report = build_table_candidates("Покажи таблицы", groups)
    assert report["sampled_candidates"] == 100 and report["partial_example_count"] > 0
    for row in descriptions(candidates):
        assert row["partial_examples"] == [0, 1, 2]
        for original, excerpt in zip(values, row["examples"]):
            assert original.startswith(excerpt) and 0 < len(excerpt) < len(original)
    shared, _ = TableSelectionService._input("Покажи таблицы", candidates)
    assert len(shared) <= 40_000


def test_1500_per_description_bound_and_no_silent_whole_row_claim():
    candidates, report = build_table_candidates("List choices", [group(values=("A" * 2000, "B" * 2000, "C" * 2000))])
    description = descriptions(candidates)[0]
    assert len(candidates[0]["description"]) <= 1500
    assert description["partial_examples"] == [0, 1, 2] and report["preview_only"]
    assert "complete" not in description and "relation" not in description


def test_prefix_budget_reduction_is_uniform_and_never_drops_ids():
    groups = [group(f"{index:064x}", title="T" * 300) for index in range(100)]
    candidates, report = build_table_candidates("Q" * 12_000, groups)
    assert len(candidates) == 100 and report["title_prefix_chars"] < 48
    assert all(row["title"] == "T" * report["title_prefix_chars"] for row in descriptions(candidates))
    assert len(TableSelectionService._input("Q" * 12_000, candidates)[0]) <= 40_000


def test_irreducible_question_or_identity_budget_fails_instead_of_dropping_candidates():
    with pytest.raises(TableNavigationInputError, match="irreducible_input_budget"):
        build_table_candidates("Q" * 39_900, [group(f"{index:064x}") for index in range(100)])
    with pytest.raises(TableNavigationInputError, match="irreducible_input_budget"):
        build_table_candidates("Q", [group("id" * 19_999)])


@pytest.mark.parametrize("change", ["duplicate", "none", "101", "empty", "question", "description_budget", "request_budget"])
def test_invalid_input_is_rejected_without_mutating_groups(change):
    groups = [group()]
    question, kwargs = "Q", {}
    if change == "duplicate":
        groups *= 2
    elif change == "none":
        groups[0]["id"] = "none"
    elif change == "101":
        groups = [group(str(index)) for index in range(101)]
    elif change == "empty":
        groups[0]["text"] = ""
    elif change == "question":
        question = " "
    elif change == "description_budget":
        kwargs["max_description_chars"] = 1501
    else:
        kwargs["max_input_chars"] = 40_001
    original = deepcopy(groups)
    with pytest.raises(TableNavigationInputError):
        build_table_candidates(question, groups, **kwargs)
    assert groups == original


def test_no_header_inference_or_semantic_relationship_for_plain_literal_rows():
    one = group(values=("Same", "Same", "Same"))
    one["text"] = "First literal row\nSecond literal row\nThird literal row\nFourth literal row"
    candidates, _ = build_table_candidates("List options", [one])
    row = descriptions(candidates)[0]
    assert row["row0"] == "First literal row"
    assert row["examples"] == ["Second literal row", "Third literal row", "Fourth literal row"]
    assert "header" not in row and "relation" not in row


def test_stable_serialization_and_exact_repetition_handling():
    groups = [group(values=("First", "Second", "Third"))]
    first = build_table_candidates("List options", groups)
    assert first == build_table_candidates("List options", groups)
    assert "Repeated type" not in descriptions(first[0])[0]["examples"]


def test_piecewise_budget_search_matches_exhaustive_short_value_oracle():
    values = ["a" * 4, "b" * 8, "c" * 15]
    base = {"examples": [], "preview": True, "truncated": True}

    def encoded(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    baseline_cost = len(encoded(encoded(base)))
    for allowance in range(0, 120, 3):
        feasible = []
        for cap in range(16):
            candidate = {**base, "examples": [value[:cap] for value in values] if cap else []}
            partial = [index for index, value in enumerate(values) if len(value) > cap] if cap else []
            if partial:
                candidate["partial_examples"] = partial
            if len(encoded(encoded(candidate))) - baseline_cost <= allowance:
                feasible.append(cap)
        result = json.loads(table_navigation._with_examples(base, values, allowance, 1500))
        actual_cap = len(result["examples"][-1]) if result["examples"] else 0
        assert actual_cap == max(feasible)
