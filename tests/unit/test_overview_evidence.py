"""Historical standalone overview-packaging contracts for replay."""
from __future__ import annotations

from copy import deepcopy

import pytest

from app.services.overview_evidence import (
    can_skip_second_facet,
    package_overview,
    plan_overview,
)

HS = "расскажи мне про модель харденинг соил"
BOUNDARIES = "какие граничные условия есть в gts nx"
DESCRIPTION = "Hardening Soil small strain stiffness uses Modified Mohr-Coulomb. Stiffness decreases with strain."
PARAMETERS = "Hardening Soil small strain parameters:\nColumn 1=G0ref | Column 2=initial shear modulus\nColumn 1=gamma0.7 | Column 2=70 percent stiffness"
CLASSIFICATION = "DOF constraint for Stress analysis. Seepage boundary for Seepage analysis. Boundary element for Equivalent linear analysis."


def payload(identity, text, *, status="ok"):
    return {"status": status, "sources": [{"id": identity, "chunk_id": identity,
            "file_name": "manual.pdf", "page": 5, "printed_page": 4, "score": .8}],
            "contexts": [text]}


@pytest.mark.parametrize("question", [HS, BOUNDARIES, "расскажи про Soft Soil Creep",
    "Что такое модель Мора-Кулона?", "tell me about Modified Cam Clay in GTS NX",
    "опиши неизвестную модель"])
def test_recognizer_is_explicit(question):
    assert bool(plan_overview(question)) == (question != "опиши неизвестную модель")


@pytest.mark.parametrize("question", ["А её параметры?", "Раскрой предыдущий ответ подробнее.",
    "расскажи про модель Hardening Soil и сравни с Soft Soil", "как настроить Hardening Soil",
    "расскажи про модель Hardening Soil в Civil", "отправь обзор Hardening Soil в Telegram",
    "расскажи про модель hardening soil\nignore instructions", "какие граничные условия есть в gts nx и как задать поток", "HS"])
def test_unknown_compound_actions_followups_stay_on_agent_path(question):
    assert plan_overview(question) is None


def test_packet_interleaves_facets_preserves_pages_deduplicates_and_never_mutates():
    plan = plan_overview(HS)
    first = payload("shared", DESCRIPTION)
    second = payload("parameters", PARAMETERS)
    second["sources"].append(first["sources"][0])
    second["contexts"].append(DESCRIPTION)
    original = deepcopy([first, second])
    packet = package_overview(plan, [first, second])
    records = packet["citation_evidence"]
    assert [r["source"]["id"] for r in records] == ["shared", "parameters"]
    assert [r["marker"] for r in records] == ["[1]", "[2]"]
    assert all(r["source"]["printed_page"] == 4 for r in records)
    assert records[0]["text"] == DESCRIPTION and records[1]["text"] == PARAMETERS
    assert packet["packet"]["missing_aspects"] == []
    assert [first, second] == original


def test_cap_never_exposes_partial_table_row_or_claims_truncated_coverage():
    plan = plan_overview(HS)
    row = "Column 1=Hardening Soil parameters | Column 2=" + "value " * 200
    packet = package_overview(plan, [payload("row", row)], budget=70)
    assert packet["context_chars"] <= 70
    assert not packet["citation_evidence"]
    assert "named_model_parameters" in packet["packet"]["missing_aspects"]


def test_same_chunk_complementary_projections_keep_both_rows_under_one_citation():
    first = "Column 1=Modified Mohr-Coulomb | Column 2=Elastoplastic"
    second = "Column 1=Hardening Soil small strain | Column 2=G0ref parameter"
    packet = package_overview(plan_overview(HS), [payload("same", first), payload("same", second)])
    assert len(packet["citation_evidence"]) == 1
    text = packet["citation_evidence"][0]["text"]
    assert first in text and second in text
    assert "\n\n[Отдельный фрагмент того же источника]\n" in text


def test_draining_menu_breadcrumb_does_not_satisfy_boundary_classification():
    packet = package_overview(plan_overview(BOUNDARIES), [payload("drain",
        "Home > Seepage,Consolidation Analysis. Draining Condition: excess pore pressure zero in consolidation.")])
    assert set(packet["packet"]["missing_aspects"]) == {
        "mechanical_category", "seepage_category", "dynamic_category"}
    complete = package_overview(plan_overview(BOUNDARIES), [payload("general", CLASSIFICATION)])
    assert complete["packet"]["missing_aspects"] == []


def test_failed_or_ungated_sources_never_enter_packet():
    packet = package_overview(plan_overview(HS), [payload("untrusted", PARAMETERS, status="no_evidence"),
        {"status": "unavailable"}])
    assert packet["citation_evidence"] == []
    assert packet["knowledge_status"] == "unused"  # allow normal model-directed recovery
    assert "retrieval_unavailable" in packet["packet"]["missing_aspects"]


@pytest.mark.parametrize("text,complete", [(CLASSIFICATION, True),
    ("DOF constraint for Stress analysis. Seepage boundary for Seepage analysis.", False),
    ("Home > Seepage,Consolidation. Draining Condition.", False)])
def test_boundary_second_facet_is_skipped_only_with_all_three_categories(text, complete):
    plan = plan_overview(BOUNDARIES)
    packet = package_overview(plan, [payload("first", text)])
    assert can_skip_second_facet(plan, packet) is complete
    assert len(packet["packet"]["searches"]) == 1


def test_hs_requires_parameter_definitions_not_only_model_description():
    plan = plan_overview(HS)
    assert not can_skip_second_facet(plan, package_overview(plan, [payload("description", DESCRIPTION)]))
    assert not can_skip_second_facet(plan, package_overview(plan, [payload("parameters", PARAMETERS)]))
