"""Historical standalone HSS evidence contracts, retained for replay."""
from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from app.services.overview_evidence import plan_overview
from app.services.verified_overview import PILOT_SUBJECT, prepare_verified_overview

QUESTION = "Расскажи про модель Hardening Soil small strain в GTS NX"
DESCRIPTION = "Hardening Soil small strain uses Modified Mohr-Coulomb with nonlinear stiffness."
STIFFNESS = "Column 1=E50ref | Column 2=secant modulus\nColumn 1=Eoedref | Column 2=oedometer modulus\nColumn 1=Eurref | Column 2=unloading modulus"
SMALL = "Column 1=G0ref | Column 2=initial shear modulus\nColumn 1=gamma0.7 | Column 2=strain at 70 percent stiffness"


def payload(texts=(DESCRIPTION, STIFFNESS, SMALL), *, subject=PILOT_SUBJECT, version=None):
    sources = []
    for index, text in enumerate(texts):
        identity = f"chunk-{index}"
        sources.append({"id": identity, "chunk_id": identity, "file_name": "manual.chm", "confident": True,
                        "evidence_scope": {"status": "exact", "subject": subject, "product": "gts_nx",
                            "chunk_id": identity, "file_name": "manual.chm", "document_version": version,
                            "chunk_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "normalized_span": {"start": index * 100, "end": index * 100 + len(text)},
                            "chm_member": "/Material/HSS.htm", "section_path": ["Material", "HSS.htm"],
                            "title_anchor": {"title": "Hardening Soil(small strain stiffness)",
                                             "member_sha256": "a" * 64}}})
    return {"status": "ok", "sources": sources, "contexts": list(texts)}


def test_scope_from_title_binds_parameter_table_without_model_name_and_keeps_text():
    result = payload()
    before = deepcopy(result)
    prepared = prepare_verified_overview(plan_overview(QUESTION), [result])
    assert prepared["ready"]
    assert prepared["packet"]["missing_aspects"] == []
    assert result == before
    assert [record["text"] for record in prepared["citation_evidence"]] == result["contexts"]
    assert [record["marker"] for record in prepared["citation_evidence"]] == ["[1]", "[2]", "[3]"]
    assert all(record["text_sha256"] == hashlib.sha256(record["text"].encode()).hexdigest()
               for record in prepared["citation_evidence"])


@pytest.mark.parametrize("field,value", [("status", "unknown"), ("subject", "Hardening Soil"),
    ("product", "civil"), ("chunk_id", "different"), ("chunk_sha256", "missing")])
def test_body_terms_cannot_repair_wrong_or_unverified_scope(field, value):
    result = payload()
    for source in result["sources"]:
        source["evidence_scope"][field] = value
    prepared = prepare_verified_overview(plan_overview(QUESTION), [result])
    assert not prepared["ready"] and not prepared["sources"]
    assert prepared["rejected_sources"] == 3


def test_hss_evidence_does_not_satisfy_base_hs_or_boundary_overview():
    for question in ("Расскажи про Hardening Soil", "Какие граничные условия есть в GTS NX"):
        prepared = prepare_verified_overview(plan_overview(question), [payload()])
        assert not prepared["ready"] and not prepared["sources"]


def test_version_conflict_and_retrieval_failure_block_generation():
    for results in ([payload(version="2021"), payload(version="2024")],
                    [payload(), payload(version="2021")], [payload(), {"status": "unavailable"}]):
        prepared = prepare_verified_overview(plan_overview(QUESTION), results)
        assert not prepared["ready"]
        assert prepared["packet"]["missing_aspects"]


def test_coverage_is_checked_after_budget_and_never_recovers_dropped_rows():
    prepared = prepare_verified_overview(plan_overview(QUESTION), [payload()], budget=100)
    assert not prepared["ready"]
    assert prepared["context_chars"] <= 100
    assert "stiffness_parameters" in prepared["packet"]["missing_aspects"]
